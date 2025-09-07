#!/usr/bin/env python3
# daytwo.py — Day 2: EchoKey “Recursion” → ZYZ rewrite (Qiskit 1.4), layout-aware
#
# WHAT THIS FILE SHOWS (high level)
# ---------------------------------
# 1) EchoKey operator family recap:
#    Each (traceless) local EchoKey operator is E_k^∘ = a_k · σ with a_k ∈ ℝ^3.
#    A linear combo H_EK = Σ_k c_k E_k^∘ has Pauli vector α = A^T c, where A rows are a_k^T.
#    Day-2 uses k=1 (“Recursion” in the 7-op frame): ek_rec(θ) ≡ exp(-i θ (a2 · σ)), with a2 = A[1].
#
# 2) A SYMBOLIC GATE:
#    EchoKeyRecursionGate(θ) is a 1-qubit instruction. It’s symbolic so the compiler can rewrite it.
#
# 3) THE REWRITE PASS (core contribution):
#    EchoKeyRecursionRewritePass maps each ek_rec(θ) to a Z–Y–Z Euler factorization:
#         ek_rec(θ)  →  RZ(α) · RY(β) · RZ(γ)
#    computed from the axis–angle rotation R_{n̂}(ϕ) with  n̂ = A[phys_q][1]/||·||  and  ϕ = 2θ.
#    We form U = cos(ϕ/2) I − i sin(ϕ/2) (n̂·σ) and use OneQubitEulerDecomposer("ZYZ").
#
# 4) LAYOUT AWARENESS:
#    If the pass property_set contains a layout (“final_layout” or “layout”), we map
#    the node’s logical qubit → physical index and fetch that wire’s A before computing n̂.
#
# 5) REFERENCE PATH FOR TESTS:
#    For fidelity checks, we “materialize” the original circuit by replacing ek_rec(θ) with the
#    exact 2×2 unitary UnitaryGate(exp(-i θ a2·σ)) — the ground truth. Then run ONLY our pass
#    (no layout/routing) and compare unitaries. Result: fidelities ≈ 1.
#
# 6) EXAMPLES:
#    - 1q simple (ek_rec → H around a chosen axis)
#    - 1q multi-step mixed with native rotations
#    - 2q with CX between ek_rec on different qubits
#    - 4q with per-site axes + CX chain
#    - randomized battery (1–3 qubits)
#
# NOTE:
#    This file mirrors Day-1 (Cyclicity) but selects A row 1 (the second EchoKey direction).
#    It’s drop-in parallel to echokey_cyclicity.py, keeping the same testing pattern.

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy.linalg import expm

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.library import RZGate, RYGate, RXGate, HGate, CXGate, UnitaryGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer  # Qiskit 1.4 location
from qiskit.transpiler import PassManager

# -------------------------------
# 0) “Ghetto math”: define A rows
# -------------------------------

def ek7_A_default() -> np.ndarray:
    """
    Return a baseline 7×3 coefficient matrix A, whose rows are a_k^T.
    Day-2 uses row 1 (“Recursion”). Day-1 used row 0 (“Cyclicity”).
    """
    A = np.array([
        [ 1, 0, 0],  # k=0 : cyclicity (a1)
        [ 0, 1, 0],  # k=1 : recursion (a2)
        [ 0, 0, 1],  # k=2
        [ 1, 1, 0],
        [ 0, 1, 1],
        [ 1, 0, 1],
        [ 1, 1, 1],
    ], float)
    # normalize each row to unit length (scale-agnostic)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A


def axis_from_angles(theta_xy: float, tilt_z: float) -> np.ndarray:
    """Utility: build a unit vector n̂ from planar angle + Z tilt."""
    v = np.array([np.cos(theta_xy), np.sin(theta_xy), 0.0], float) + np.array([0.0, 0.0, tilt_z], float)
    n = np.linalg.norm(v)
    return v / (n if n > 0 else 1.0)


def site_specific_A(num_qubits: int) -> Dict[int, np.ndarray]:
    """
    Produce a dict {q: A_q} so each qubit gets its own A, with row 1 (a2) on a different axis.
    This mimics attaching a physical orientation to each wire/site.
    """
    As: Dict[int, np.ndarray] = {}
    for q in range(num_qubits):
        a2 = axis_from_angles(theta_xy=2*np.pi*q/max(1, num_qubits), tilt_z=0.25*np.sin(q + 0.25))
        A = ek7_A_default().copy()
        A[1] = a2 / np.linalg.norm(a2)
        As[q] = A
    return As

# ----------------------------------
# 1) Symbolic EchoKey “Recursion” op
# ----------------------------------

class EchoKeyRecursionGate(Gate):
    """
    Symbolic single-qubit EchoKey instruction for Day-2 (row 1).
    ek_rec(θ) ≡ exp(-i θ (a2 · σ)).
    """
    def __init__(self, theta: float):
        super().__init__(name="ek_rec", num_qubits=1, params=[float(theta)])
        # symbolic — no definition matrix here (lets the pass rewrite it)
        self.definition = QuantumCircuit(1, name="ek_rec")

# --------------------------------------------------
# 2) Axis–angle → ZYZ helper (stable, exact angles)
# --------------------------------------------------

_DECOMP_ZYZ = OneQubitEulerDecomposer(basis="ZYZ")

def axis_angle_to_zyz(nhat: np.ndarray, phi: float) -> Tuple[float, float, float]:
    """
    Given axis n̂ (unit ℝ^3) and Bloch angle φ, return ZYZ Euler angles (α, β, γ)
    such that RZ(α)·RY(β)·RZ(γ) = exp(-i (φ/2) n̂·σ) up to global phase.
    """
    nx, ny, nz = nhat
    I2 = np.eye(2, dtype=complex)
    SX = np.array([[0, 1],[1, 0]], complex)
    SY = np.array([[0,-1j],[1j, 0]], complex)
    SZ = np.array([[1, 0],[0,-1 ]], complex)
    U = np.cos(phi/2) * I2 - 1j * np.sin(phi/2) * (nx*SX + ny*SY + nz*SZ)
    alpha, beta, gamma = _DECOMP_ZYZ.angles(U)  # Qiskit 1.4
    return float(alpha), float(beta), float(gamma)

# -------------------------------------------------
# 3) The rewrite pass: ek_rec(θ) → RZ–RY–RZ (ZYZ)
#    with layout-aware per-wire axis resolution
# -------------------------------------------------

class EchoKeyRecursionRewritePass(TransformationPass):
    """
    ek_rec(θ)  →  RZ(α) · RY(β) · RZ(γ)
    with axis n̂ = normalize(A[phys_q][1]) and Bloch angle φ = 2θ.
    Layout (if present) is taken from self.property_set.
    """

    def __init__(self, weights_per_qubit: Dict[int, "SiteWeights"]):
        super().__init__()
        self.W = weights_per_qubit

    def _physical_index_for_node(self, dag: DAGCircuit, node) -> int:
        """
        Return the physical qubit index for a 1q node if a layout exists; else return logical index.
        """
        logical_idx = dag.find_bit(node.qargs[0]).index
        layout_obj = self.property_set.get("final_layout") or self.property_set.get("layout")
        if layout_obj is None:
            return logical_idx
        L = getattr(layout_obj, "final_layout", None) or layout_obj
        try:
            phys = L[node.qargs[0]]  # PhysicalQubit or int
        except Exception:
            return logical_idx
        return int(getattr(phys, "index", phys))

    _DECOMP_ZYZ = OneQubitEulerDecomposer(basis="ZYZ")

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # Pauli matrices for building the exact 2x2 unitary
        SX = np.array([[0, 1],[1, 0]], complex)
        SY = np.array([[0,-1j],[1j, 0]], complex)
        SZ = np.array([[1, 0],[0,-1 ]], complex)
        I2 = np.eye(2, dtype=complex)

        for node in list(dag.op_nodes()):
            if isinstance(node.op, EchoKeyRecursionGate):
                theta = float(node.op.params[0])

                # (1) resolve physical wire if a layout is present
                phys_q = self._physical_index_for_node(dag, node)

                # (2) get per-wire A and unit axis n̂ (row 1 → a2)
                A = self.W.get(phys_q, SiteWeights(ek7_A_default())).A
                a2 = A[1]
                if not np.isfinite(a2).all() or np.linalg.norm(a2) < 1e-15:
                    raise ValueError(f"A[1] invalid/degenerate for qubit {phys_q}")
                nhat = a2 / np.linalg.norm(a2)

                # (3) exact SU(2) unitary for ek_rec(θ): U = cos(θ) I - i sin(θ) (n̂·σ)
                U = np.cos(theta) * I2 - 1j * np.sin(theta) * (nhat[0]*SX + nhat[1]*SY + nhat[2]*SZ)

                # (4) synthesize ZYZ circuit exactly from U
                rep_qc = self._DECOMP_ZYZ(U)  # returns a 1-qubit circuit over {RZ, RY}
                dag.substitute_node_with_dag(node, circuit_to_dag(rep_qc))

        return dag

# -------------------------------
# 4) Per-qubit weights container
# -------------------------------

@dataclass
class SiteWeights:
    A: np.ndarray  # (7×3) for that qubit/wire

# -------------------------------------------------------------------
# 5) Reference path: materialize ek_rec(θ) → exact UnitaryGate(U₂×₂)
#    (used only to build a ground-truth circuit for fidelity checks)
# -------------------------------------------------------------------

def materialize_ek_ops(qc: QuantumCircuit, weights: Dict[int, SiteWeights]) -> QuantumCircuit:
    """
    Walk the original circuit and replace each EchoKeyRecursionGate(θ) with the exact
    2×2 UnitaryGate(U) where U = exp(-i θ a2·σ). We keep everything else unchanged.
    """
    out = QuantumCircuit(qc.num_qubits)
    for creg in qc.cregs:
        out.add_register(creg)

    SX = np.array([[0, 1],[1, 0]], complex)
    SY = np.array([[0,-1j],[1j, 0]], complex)
    SZ = np.array([[1, 0],[0,-1 ]], complex)

    for inst in qc.data:  # CircuitInstruction in Qiskit 1.4
        op, qubits, clbits = inst.operation, inst.qubits, inst.clbits
        if isinstance(op, EchoKeyRecursionGate):
            theta = float(op.params[0])
            q_index = qc.find_bit(qubits[0]).index  # logical (reference path)
            A = weights.get(q_index, SiteWeights(ek7_A_default())).A
            a2 = A[1] / np.linalg.norm(A[1])
            H = theta * (a2[0]*SX + a2[1]*SY + a2[2]*SZ)
            U = expm(-1j * H)
            out.append(UnitaryGate(U, label="ek_rec(materialized)"), [qubits[0]])
        else:
            out.append(op, qubits, clbits)
    return out

# -------------------------------------------------------
# 6) Plumbing for unitary extraction & “pass-only” compile
# -------------------------------------------------------

def unitary(circ: QuantumCircuit) -> np.ndarray:
    """Exact unitary of a (small) circuit (includes global phase)."""
    return Operator(circ).data


def fid(U: np.ndarray, V: np.ndarray) -> float:
    """Global-phase-insensitive fidelity proxy: |Tr(U†V)| / 2^n."""
    d = U.shape[0]
    return float(abs(np.trace(U.conj().T @ V)) / d)


def compile_with_pass_only(qc: QuantumCircuit, weights: Dict[int, SiteWeights]) -> QuantumCircuit:
    """
    Day-2 proof mode: run ONLY our rewrite pass (no layout/routing), so wire order
    stays stable for apples-to-apples unitary comparison.
    """
    pm = PassManager([EchoKeyRecursionRewritePass(weights)])
    return pm.run(qc)

# ---------------------------
# 7) Examples (mirrors Day-1)
# ---------------------------

def ex1_simple() -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    qc = QuantumCircuit(1)
    qc.append(EchoKeyRecursionGate(0.37), [0])
    qc.append(HGate(), [0])
    weights = {0: SiteWeights(A=ek7_A_default())}
    return ("[Ex1] 1q simple ek_rec → H", qc, weights)


def ex2_sequence() -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    qc = QuantumCircuit(1)
    qc.append(RZGate(0.11), [0])
    qc.append(EchoKeyRecursionGate(-0.42), [0])
    qc.append(RYGate(0.23), [0])
    qc.append(EchoKeyRecursionGate(0.80), [0])
    qc.append(RXGate(-0.31), [0])
    weights = {0: SiteWeights(A=ek7_A_default())}
    return ("[Ex2] 1q multi-step (RZ, ek_rec, RY, ek_rec, RX)", qc, weights)


def ex3_two_qubit() -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    qc = QuantumCircuit(2)
    qc.append(HGate(), [0])
    qc.append(EchoKeyRecursionGate(0.50), [0])
    qc.append(CXGate(), [0, 1])
    qc.append(EchoKeyRecursionGate(-0.25), [1])
    qc.append(RZGate(0.40), [1])
    weights = {0: SiteWeights(ek7_A_default()), 1: SiteWeights(ek7_A_default())}
    return ("[Ex3] 2q CX between ek_rec ops", qc, weights)


def ex4_multiqubit_per_site_axes(n: int = 4) -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    qc = QuantumCircuit(n)
    # give each site its own a2
    As = site_specific_A(n)
    for q in range(n):
        qc.append(EchoKeyRecursionGate(0.17*(q+1)), [q])
        if q < n-1:
            qc.append(CXGate(), [q, q+1])
    weights = {q: SiteWeights(A=As[q]) for q in range(n)}
    return (f"[Ex4] {n}q per-site a2 + CX chain", qc, weights)


def ex5_random_battery(m: int = 8, seed: int = 42) -> List[Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]]:
    rng = np.random.default_rng(seed)
    out: List[Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]] = []
    for i in range(m):
        n = int(rng.integers(1, 4))  # 1–3 qubits
        L = int(rng.integers(3, 8))  # 3–7 ops
        qc = QuantumCircuit(n)
        As = site_specific_A(n)
        for _ in range(L):
            sel = int(rng.integers(0, 4))
            if sel == 0:
                q = int(rng.integers(0, n)); qc.append(EchoKeyRecursionGate(float(rng.normal()*0.7)), [q])
            elif sel == 1:
                q = int(rng.integers(0, n)); qc.append(RYGate(float(rng.normal()*0.5)), [q])
            elif sel == 2:
                q = int(rng.integers(0, n)); qc.append(RZGate(float(rng.normal()*0.5)), [q])
            else:
                if n >= 2:
                    q1 = int(rng.integers(0, n-1)); qc.append(CXGate(), [q1, q1+1])
        weights = {q: SiteWeights(A=As[q]) for q in range(n)}
        out.append((f"[Ex5.{i+1}] random n={n}, L={L}", qc, weights))
    return out

# ---------------------------
# 8) Runner with fidelity checks
# ---------------------------

def run_and_check(label: str, qc: QuantumCircuit, weights: Dict[int, SiteWeights], show_circuits=False):
    compiled = compile_with_pass_only(qc, weights)   # pass-only for clean proof
    qc_ref  = materialize_ek_ops(qc, weights)        # ground truth (exact UnitaryGate for ek_rec)
    U_in, U_out = unitary(qc_ref), unitary(compiled)
    F = fid(U_in, U_out)
    print(f"{label:32s}  fidelity = {F:.12f}")
    if show_circuits:
        print("\nOriginal (symbolic):")
        print(qc.draw(fold=-1))
        print("\nMaterialized original (ground truth):")
        print(qc_ref.draw(fold=-1))
        print("\nRewritten by pass (ZYZ basis):")
        print(compiled.draw(fold=-1))


def main():
    print("=== Day 2: Recursion — ZYZ rewrite (Qiskit 1.4) + layout-aware ===")
    # Example 1: simple
    run_and_check(*ex1_simple(), show_circuits=True)
    # Example 2: multi-step single-qubit
    run_and_check(*ex2_sequence(), show_circuits=False)
    # Example 3: 2-qubit with CX in the middle
    run_and_check(*ex3_two_qubit(), show_circuits=False)
    # Example 4: 4-qubit with per-site recursion axes + entanglers
    run_and_check(*ex4_multiqubit_per_site_axes(4), show_circuits=False)
    # Example 5: randomized battery
    for label, qc, weights in ex5_random_battery(m=8, seed=42):
        run_and_check(label, qc, weights, show_circuits=False)


if __name__ == "__main__":
    main()
