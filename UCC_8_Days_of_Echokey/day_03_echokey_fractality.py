#!/usr/bin/env python3
# daythree.py — Day 3: EchoKey “Fractality” → ZYZ rewrite (with Z‑axis fast path), layout‑aware
#
# WHAT THIS FILE SHOWS (high level)
# ---------------------------------
# 1) EchoKey operator family recap:
#    Each (traceless) local EchoKey operator is E_k^∘ = a_k · σ with a_k ∈ ℝ^3.
#    A linear combo H_EK = Σ_k c_k E_k^∘ has Pauli vector α = A^T c, where A rows are a_k^T.
#    Day‑3 uses k=2 (“Fractality” in the 7‑op frame): ek_frac(θ) ≡ exp(−i θ (a3 · σ)), with a3 = A[2].
#
# 2) A SYMBOLIC GATE:
#    EchoKeyFractalityGate(θ) is a 1‑qubit instruction. It’s symbolic so the compiler can rewrite it.
#
# 3) THE REWRITE PASS (core contribution):
#    EchoKeyFractalityRewritePass maps ek_frac(θ) to:
#       • FAST PATH (axis ≈ ±Z):    RZ(±2θ)
#       • GENERAL (arbitrary axis): RZ(α) · RY(β) · RZ(γ)   via exact axis–angle synthesis
#    We compute the axis–angle rotation R_{n̂}(φ) with  n̂ = normalize(A[phys_q][2])  and  φ = 2θ.
#    Materialize U = cos(φ/2) I − i sin(φ/2) (n̂·σ) and use OneQubitEulerDecomposer("ZYZ").
#
# 4) LAYOUT AWARENESS:
#    If the pass property_set contains a layout (“final_layout” or “layout”), we map
#    the node’s logical qubit → physical index and fetch that wire’s A before computing n̂.
#
# 5) REFERENCE PATH FOR TESTS:
#    We “materialize” the original circuit by replacing ek_frac(θ) with the exact 2×2 unitary
#    UnitaryGate(exp(−i θ a3·σ)) — the ground truth. Then run ONLY our pass and compare unitaries.
#
# 6) EXAMPLES:
#    - 1q pure Z‑axis fast path  (a3 = Z) → ek_frac → RZ
#    - 1q sequence with mixed native rotations
#    - 2q with CX between ek_frac ops
#    - 4q per‑site frames where some wires tilt a3 off Z (to show general ZYZ path)
#    - randomized battery (1–3 qubits)
#
# NOTE:
#    This file mirrors Day‑1/2 but selects A row 2 (the third EchoKey direction). When a3 = (0,0,±1),
#    exp(−i θ σ_z) equals RZ(2θ); we detect this and emit a single native gate.

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
    Day‑3 uses row 2 (“Fractality”). Rows are normalized to unit length.
    """
    A = np.array([
        [ 1, 0, 0],  # k=0 : cyclicity (X)
        [ 0, 1, 0],  # k=1 : recursion (Y)
        [ 0, 0, 1],  # k=2 : fractality (Z)
        [ 1, 1, 0],
        [ 0, 1, 1],
        [ 1, 0, 1],
        [ 1, 1, 1],
    ], float)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A


def axis_from_angles(theta_xy: float, tilt_z: float) -> np.ndarray:
    """Utility: build a unit vector n̂ from planar angle + Z tilt."""
    v = np.array([np.cos(theta_xy), np.sin(theta_xy), tilt_z], float)
    n = np.linalg.norm(v)
    return v / (n if n > 0 else 1.0)


def site_specific_A(num_qubits: int, tilt_every:int=2) -> Dict[int, np.ndarray]:
    """
    Produce a dict {q: A_q} so each qubit gets its own A. By default a3 = Z for all wires
    (fast path). To demonstrate the general ZYZ path, we optionally tilt a3 on every
    `tilt_every`‑th wire.
    """
    As: Dict[int, np.ndarray] = {}
    base = ek7_A_default()
    for q in range(num_qubits):
        A = base.copy()
        if tilt_every > 0 and (q % tilt_every == 1):  # tilt some a3’s slightly off Z
            a3 = axis_from_angles(theta_xy=2*np.pi*q/max(1, num_qubits), tilt_z=1.0)
            A[2] = a3 / np.linalg.norm(a3)
        As[q] = A
    return As

# ----------------------------------
# 1) Symbolic EchoKey “Fractality” op
# ----------------------------------

class EchoKeyFractalityGate(Gate):
    """
    Symbolic single‑qubit EchoKey instruction for Day‑3 (row 2).
    ek_frac(θ) ≡ exp(−i θ (a3 · σ)).
    """
    def __init__(self, theta: float):
        super().__init__(name="ek_frac", num_qubits=1, params=[float(theta)])
        # symbolic — no definition matrix here (lets the pass rewrite it)
        self.definition = QuantumCircuit(1, name="ek_frac")

# --------------------------------------------------
# 2) Axis–angle → ZYZ helper (stable, exact angles)
# --------------------------------------------------

_DECOMP_ZYZ = OneQubitEulerDecomposer(basis="ZYZ")

def axis_angle_to_zyz(nhat: np.ndarray, phi: float) -> Tuple[float, float, float]:
    """
    Given axis n̂ (unit ℝ^3) and Bloch angle φ, return ZYZ Euler angles (α, β, γ)
    such that RZ(α)·RY(β)·RZ(γ) = exp(−i (φ/2) n̂·σ) up to global phase.
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
# 3) The rewrite pass: ek_frac(θ) → RZ (fast) or ZYZ
#    with layout‑aware per‑wire axis resolution
# -------------------------------------------------

class EchoKeyFractalityRewritePass(TransformationPass):
    """
    ek_frac(θ)  →  RZ(±2θ)   if axis ≈ ±Z
                 →  RZ(α) · RY(β) · RZ(γ)   otherwise
    with axis n̂ = normalize(A[phys_q][2]) and Bloch angle φ = 2θ.
    Layout (if present) is read from self.property_set.
    """

    def __init__(self, weights_per_qubit: Dict[int, "SiteWeights"], z_eps: float = 1e-12):
        super().__init__()
        self.W = weights_per_qubit
        self.z_eps = float(z_eps)
        self._decomp = OneQubitEulerDecomposer(basis="ZYZ")

    def _physical_index_for_node(self, dag: DAGCircuit, node) -> int:
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

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        SX = np.array([[0, 1],[1, 0]], complex)
        SY = np.array([[0,-1j],[1j, 0]], complex)
        SZ = np.array([[1, 0],[0,-1 ]], complex)
        I2 = np.eye(2, dtype=complex)

        for node in list(dag.op_nodes()):
            if isinstance(node.op, EchoKeyFractalityGate):
                theta = float(node.op.params[0])
                phys_q = self._physical_index_for_node(dag, node)

                A = self.W.get(phys_q, SiteWeights(ek7_A_default())).A
                a3 = A[2]
                if not np.isfinite(a3).all() or np.linalg.norm(a3) < 1e-15:
                    raise ValueError(f"A[2] invalid/degenerate for qubit {phys_q}")
                nhat = a3 / np.linalg.norm(a3)

                # FAST PATH: axis ≈ +Z or −Z  →  RZ(±2θ)
                if abs(nhat[0]) < self.z_eps and abs(nhat[1]) < self.z_eps and abs(abs(nhat[2]) - 1.0) < self.z_eps:
                    lam = 2.0 * theta * (1.0 if nhat[2] >= 0 else -1.0)
                    rep_qc = QuantumCircuit(1)
                    rep_qc.append(RZGate(lam), [0])
                    dag.substitute_node_with_dag(node, circuit_to_dag(rep_qc))
                    continue

                # GENERAL PATH: exact SU(2) → ZYZ
                U = np.cos(theta) * I2 - 1j * np.sin(theta) * (nhat[0]*SX + nhat[1]*SY + nhat[2]*SZ)
                rep_qc = self._decomp(U)
                dag.substitute_node_with_dag(node, circuit_to_dag(rep_qc))

        return dag

# -------------------------------
# 4) Per‑qubit weights container
# -------------------------------

@dataclass
class SiteWeights:
    A: np.ndarray  # (7×3) for that qubit/wire

# -------------------------------------------------------------------
# 5) Reference path: materialize ek_frac(θ) → exact UnitaryGate(U₂×₂)
# -------------------------------------------------------------------

def materialize_ek_ops(qc: QuantumCircuit, weights: Dict[int, SiteWeights]) -> QuantumCircuit:
    out = QuantumCircuit(qc.num_qubits)
    for creg in qc.cregs:
        out.add_register(creg)

    SX = np.array([[0, 1],[1, 0]], complex)
    SY = np.array([[0,-1j],[1j, 0]], complex)
    SZ = np.array([[1, 0],[0,-1 ]], complex)

    for inst in qc.data:  # CircuitInstruction in Qiskit 1.4
        op, qubits, clbits = inst.operation, inst.qubits, inst.clbits
        if isinstance(op, EchoKeyFractalityGate):
            theta = float(op.params[0])
            q_index = qc.find_bit(qubits[0]).index  # logical (reference path)
            A = weights.get(q_index, SiteWeights(ek7_A_default())).A
            a3 = A[2] / np.linalg.norm(A[2])
            H = theta * (a3[0]*SX + a3[1]*SY + a3[2]*SZ)
            U = expm(-1j * H)
            out.append(UnitaryGate(U, label="ek_frac(materialized)"), [qubits[0]])
        else:
            out.append(op, qubits, clbits)
    return out

# -------------------------------------------------------
# 6) Plumbing for unitary extraction & “pass‑only” compile
# -------------------------------------------------------

def unitary(circ: QuantumCircuit) -> np.ndarray:
    return Operator(circ).data


def fid(U: np.ndarray, V: np.ndarray) -> float:
    d = U.shape[0]
    return float(abs(np.trace(U.conj().T @ V)) / d)


def compile_with_pass_only(qc: QuantumCircuit, weights: Dict[int, SiteWeights]) -> QuantumCircuit:
    pm = PassManager([EchoKeyFractalityRewritePass(weights)])
    return pm.run(qc)

# ---------------------------
# 7) Examples (mirrors Day‑1/2)
# ---------------------------

def ex1_simple_fastpath() -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    qc = QuantumCircuit(1)
    qc.append(EchoKeyFractalityGate(0.40), [0])
    qc.append(HGate(), [0])
    weights = {0: SiteWeights(A=ek7_A_default())}
    return ("[Ex1] 1q ek_frac (Z‑axis fast path) → H", qc, weights)


def ex2_sequence() -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    qc = QuantumCircuit(1)
    qc.append(RXGate(0.11), [0])
    qc.append(EchoKeyFractalityGate(-0.42), [0])
    qc.append(RYGate(0.23), [0])
    qc.append(EchoKeyFractalityGate(0.80), [0])
    qc.append(RZGate(-0.31), [0])
    weights = {0: SiteWeights(A=ek7_A_default())}
    return ("[Ex2] 1q sequence (RX, ek_frac, RY, ek_frac, RZ)", qc, weights)


def ex3_two_qubit() -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    qc = QuantumCircuit(2)
    qc.append(HGate(), [0])
    qc.append(EchoKeyFractalityGate(0.50), [0])
    qc.append(CXGate(), [0, 1])
    qc.append(EchoKeyFractalityGate(-0.25), [1])
    qc.append(RYGate(0.40), [1])
    weights = {0: SiteWeights(ek7_A_default()), 1: SiteWeights(ek7_A_default())}
    return ("[Ex3] 2q CX between ek_frac ops", qc, weights)


def ex4_multiqubit_mixed_axes(n: int = 4) -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    qc = QuantumCircuit(n)
    # give each site its own a3; tilt every second site to trigger general ZYZ path
    As = site_specific_A(n, tilt_every=2)
    for q in range(n):
        qc.append(EchoKeyFractalityGate(0.17*(q+1)), [q])
        if q < n-1:
            qc.append(CXGate(), [q, q+1])
    weights = {q: SiteWeights(A=As[q]) for q in range(n)}
    return (f"[Ex4] {n}q per‑site a3 (mixed fast+general) + CX chain", qc, weights)


def ex5_random_battery(m: int = 8, seed: int = 43) -> List[Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]]:
    rng = np.random.default_rng(seed)
    out: List[Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]] = []
    for i in range(m):
        n = int(rng.integers(1, 4))  # 1–3 qubits
        L = int(rng.integers(3, 8))  # 3–7 ops
        qc = QuantumCircuit(n)
        As = site_specific_A(n, tilt_every=2)
        for _ in range(L):
            sel = int(rng.integers(0, 4))
            if sel == 0:
                q = int(rng.integers(0, n)); qc.append(EchoKeyFractalityGate(float(rng.normal()*0.7)), [q])
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
    compiled = compile_with_pass_only(qc, weights)   # pass‑only for clean proof
    qc_ref  = materialize_ek_ops(qc, weights)        # ground truth (exact UnitaryGate for ek_frac)
    U_in, U_out = unitary(qc_ref), unitary(compiled)
    F = fid(U_in, U_out)
    print(f"{label:44s}  fidelity = {F:.12f}")
    if show_circuits:
        print("\nOriginal (symbolic):")
        print(qc.draw(fold=-1))
        print("\nMaterialized original (ground truth):")
        print(qc_ref.draw(fold=-1))
        print("\nRewritten by pass (RZ fast path / ZYZ):")
        print(compiled.draw(fold=-1))


def main():
    print("=== Day 3: Fractality — ZYZ rewrite with Z‑axis fast path (Qiskit 1.4) ===")
    # Example 1: pure Z‑axis fast path
    run_and_check(*ex1_simple_fastpath(), show_circuits=True)
    # Example 2: multi‑step single‑qubit
    run_and_check(*ex2_sequence(), show_circuits=False)
    # Example 3: 2‑qubit with CX in the middle
    run_and_check(*ex3_two_qubit(), show_circuits=False)
    # Example 4: mixed per‑site a3 (some tilted → general ZYZ)
    run_and_check(*ex4_multiqubit_mixed_axes(4), show_circuits=False)
    # Example 5: randomized battery
    for label, qc, weights in ex5_random_battery(m=8, seed=43):
        run_and_check(label, qc, weights, show_circuits=False)


if __name__ == "__main__":
    main()
