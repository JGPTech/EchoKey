#!/usr/bin/env python3
"""
8 Days of EchoKey — Day 6: Diagonality (XZ) → ZYZ rewrite, layout‑aware
CC0‑1.0 — Public Domain

WHAT THIS FILE SHOWS (high level)
---------------------------------
1) EchoKey recap: each local generator is E_k^∘ = a_k · σ with a_k ∈ ℝ^3, ||a_k||=1 (rows of A).
2) Day‑6 selects the 6th direction (code index k=5), the XZ‑diagonal: a6 ≈ (1,0,1)/√2.
   Define ek_diagxz(θ) := exp(−i θ (a6 · σ)).
3) We provide a symbolic gate + a layout‑aware rewrite pass:
      ek_diagxz(θ)  →  RZ(α) · RY(β) · RZ(γ)
   where (α,β,γ) are from exact SU(2) synthesis of U(2θ, a6). No approximations; fidelity ≈ 1.
4) A materializer replaces the symbolic gate with its exact 2×2 unitary (ground truth) so we can
   compare unitaries (phase‑insensitive overlap) with the pass‑only rewrite.
5) Examples: single‑qubit, sequences, two‑qubit with CX, per‑site frames (tilting a6), random battery.

MATH WALKTHROUGH (concise)
--------------------------
Axis–angle form: U(φ, n̂) = cos(φ/2) I − i sin(φ/2) (n̂·σ), with φ=2θ and n̂=a6.
ZYZ completeness: ∀U∈SU(2), ∃ α,β,γ s.t. U ≍ RZ(α) RY(β) RZ(γ) (≍ up to global phase).
Rewrite (per gate): resolve physical wire p via layout → use A^(p)[5] as a6 → synthesize ZYZ.

Run me:
  $ python echokey_diagonality_xz.py
"""
from __future__ import annotations
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
from qiskit.synthesis import OneQubitEulerDecomposer  # Qiskit 1.4
from qiskit.transpiler import PassManager

# ---------------------------------
# 0) Frames A and small utilities
# ---------------------------------

def ek7_A_default() -> np.ndarray:
    """Baseline 7×3 matrix A; rows are unit vectors a_k^T. Row 6 (index 5) is XZ."""
    A = np.array([
        [ 1, 0, 0],  # k=0 : X
        [ 0, 1, 0],  # k=1 : Y
        [ 0, 0, 1],  # k=2 : Z
        [ 1, 1, 0],  # k=3 : XY diagonal
        [ 0, 1, 1],  # k=4 : YZ diagonal
        [ 1, 0, 1],  # k=5 : XZ diagonal  ← Day 6
        [ 1, 1, 1],  # k=6 : XYZ body diagonal
    ], dtype=float)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A


def axis_from_components(x: float, y: float, z: float) -> np.ndarray:
    v = np.array([x, y, z], float)
    n = np.linalg.norm(v)
    return v / (n if n > 0 else 1.0)


def site_specific_A(num_qubits: int, tilt_a6_every:int=2) -> Dict[int, np.ndarray]:
    """Give each wire its own A^(p); optionally tilt the a6 row slightly toward Y on every
    `tilt_a6_every`‑th wire to exercise the general ZYZ path under layout‑aware selection.
    """
    As: Dict[int, np.ndarray] = {}
    base = ek7_A_default()
    for p in range(num_qubits):
        A = base.copy()
        if tilt_a6_every > 0 and (p % tilt_a6_every == 1):
            # start from XZ (1,0,1), add a tiny Y‑component to move it off the XZ plane
            a6 = axis_from_components(x=1.0, y=0.20, z=1.0)
            A[5] = a6
        As[p] = A
    return As

# ----------------------------------
# 1) Symbolic Day‑6 gate (k=5)
# ----------------------------------

class EchoKeyDiagonalityXZGate(Gate):
    """Symbolic ek_diagxz(θ) ≡ exp(−i θ (a6 · σ)), where a6 is row 5 (index 5) of A."""
    def __init__(self, theta: float):
        super().__init__(name="ek_diagxz", num_qubits=1, params=[float(theta)])
        self.definition = QuantumCircuit(1, name="ek_diagxz")  # symbolic placeholder

# -------------------------------------------------
# 2) The rewrite pass: ek_diagxz(θ) → RZ · RY · RZ
#    (layout‑aware axis resolution)
# -------------------------------------------------

@dataclass
class SiteWeights:
    A: np.ndarray  # 7×3 for that physical wire

class EchoKeyDiagonalityXZRewritePass(TransformationPass):
    def __init__(self, weights_per_qubit: Dict[int, SiteWeights]):
        super().__init__()
        self.W = weights_per_qubit
        self._decomp = OneQubitEulerDecomposer(basis="ZYZ")

    def _physical_index_for_node(self, dag: DAGCircuit, node) -> int:
        logical_idx = dag.find_bit(node.qargs[0]).index
        layout_obj = self.property_set.get("final_layout") or self.property_set.get("layout")
        if layout_obj is None:
            return logical_idx
        L = getattr(layout_obj, "final_layout", None) or layout_obj
        try:
            phys = L[node.qargs[0]]
        except Exception:
            return logical_idx
        return int(getattr(phys, "index", phys))

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        SX = np.array([[0, 1],[1, 0]], complex)
        SY = np.array([[0,-1j],[1j, 0]], complex)
        SZ = np.array([[1, 0],[0,-1 ]], complex)
        I2 = np.eye(2, dtype=complex)

        for node in list(dag.op_nodes()):
            if isinstance(node.op, EchoKeyDiagonalityXZGate):
                theta = float(node.op.params[0])
                p = self._physical_index_for_node(dag, node)
                A = self.W.get(p, SiteWeights(ek7_A_default())).A
                a6 = A[5]
                if not np.isfinite(a6).all() or np.linalg.norm(a6) < 1e-15:
                    raise ValueError(f"A[5] invalid/degenerate for qubit {p}")
                nhat = a6 / np.linalg.norm(a6)

                U = np.cos(theta)*I2 - 1j*np.sin(theta)*(nhat[0]*SX + nhat[1]*SY + nhat[2]*SZ)
                rep_qc = self._decomp(U)  # 1q circuit on {RZ, RY}
                dag.substitute_node_with_dag(node, circuit_to_dag(rep_qc))
        return dag

# -------------------------------------------------------
# 3) Reference path (materialize exact 2×2 unitary)
# -------------------------------------------------------

def materialize_ek_ops(qc: QuantumCircuit, weights: Dict[int, SiteWeights]) -> QuantumCircuit:
    out = QuantumCircuit(qc.num_qubits)
    for creg in qc.cregs:
        out.add_register(creg)
    SX = np.array([[0, 1],[1, 0]], complex)
    SY = np.array([[0,-1j],[1j, 0]], complex)
    SZ = np.array([[1, 0],[0,-1 ]], complex)

    for inst in qc.data:
        op, qubits, clbits = inst.operation, inst.qubits, inst.clbits
        if isinstance(op, EchoKeyDiagonalityXZGate):
            theta = float(op.params[0])
            q_index = qc.find_bit(qubits[0]).index
            A = weights.get(q_index, SiteWeights(ek7_A_default())).A
            a6 = A[5] / np.linalg.norm(A[5])
            H = theta * (a6[0]*SX + a6[1]*SY + a6[2]*SZ)
            U = expm(-1j * H)
            out.append(UnitaryGate(U, label="ek_diagxz(mat)"), [qubits[0]])
        else:
            out.append(op, qubits, clbits)
    return out

# ----------------------------------------------
# 4) Unitary, fidelity, and pass‑only compilation
# ----------------------------------------------

def unitary(circ: QuantumCircuit) -> np.ndarray:
    return Operator(circ).data


def fid(U: np.ndarray, V: np.ndarray) -> float:
    d = U.shape[0]
    return float(abs(np.trace(U.conj().T @ V)) / d)


def compile_with_pass_only(qc: QuantumCircuit, weights: Dict[int, SiteWeights]) -> QuantumCircuit:
    pm = PassManager([EchoKeyDiagonalityXZRewritePass(weights)])
    return pm.run(qc)

# ---------------------------
# 5) Examples (mirrors others)
# ---------------------------

def ex1_simple() -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    qc = QuantumCircuit(1)
    qc.append(EchoKeyDiagonalityXZGate(0.40), [0])
    qc.append(HGate(), [0])
    weights = {0: SiteWeights(A=ek7_A_default())}
    return ("[Ex1] 1q ek_diagxz → H", qc, weights)


def ex2_sequence() -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    qc = QuantumCircuit(1)
    qc.append(RXGate(0.11), [0])
    qc.append(EchoKeyDiagonalityXZGate(-0.42), [0])
    qc.append(RYGate(0.23), [0])
    qc.append(EchoKeyDiagonalityXZGate(0.80), [0])
    qc.append(RZGate(-0.31), [0])
    weights = {0: SiteWeights(A=ek7_A_default())}
    return ("[Ex2] 1q sequence (RX, ek_diagxz, RY, ek_diagxz, RZ)", qc, weights)


def ex3_two_qubit() -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    qc = QuantumCircuit(2)
    qc.append(HGate(), [0])
    qc.append(EchoKeyDiagonalityXZGate(0.50), [0])
    qc.append(CXGate(), [0, 1])
    qc.append(EchoKeyDiagonalityXZGate(-0.25), [1])
    qc.append(RYGate(0.40), [1])
    weights = {0: SiteWeights(ek7_A_default()), 1: SiteWeights(ek7_A_default())}
    return ("[Ex3] 2q CX between ek_diagxz ops", qc, weights)


def ex4_multiqubit_mixed_axes(n: int = 4) -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    qc = QuantumCircuit(n)
    As = site_specific_A(n, tilt_a6_every=2)
    for q in range(n):
        qc.append(EchoKeyDiagonalityXZGate(0.17*(q+1)), [q])
        if q < n-1:
            qc.append(CXGate(), [q, q+1])
    weights = {q: SiteWeights(A=As[q]) for q in range(n)}
    return (f"[Ex4] {n}q per‑site a6 (some tilted) + CX chain", qc, weights)


def ex5_random_battery(m: int = 8, seed: int = 46) -> List[Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]]:
    rng = np.random.default_rng(seed)
    out: List[Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]] = []
    for i in range(m):
        n = int(rng.integers(1, 4))
        L = int(rng.integers(3, 8))
        qc = QuantumCircuit(n)
        As = site_specific_A(n, tilt_a6_every=2)
        for _ in range(L):
            sel = int(rng.integers(0, 4))
            if sel == 0:
                q = int(rng.integers(0, n)); qc.append(EchoKeyDiagonalityXZGate(float(rng.normal()*0.7)), [q])
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
# 6) Runner with fidelity checks
# ---------------------------

def run_and_check(label: str, qc: QuantumCircuit, weights: Dict[int, SiteWeights], show_circuits=False):
    compiled = compile_with_pass_only(qc, weights)
    qc_ref  = materialize_ek_ops(qc, weights)
    U_in, U_out = unitary(qc_ref), unitary(compiled)
    F = fid(U_in, U_out)
    print(f"{label:48s}  fidelity = {F:.12f}")
    if show_circuits:
        print("\nOriginal (symbolic):\n", qc.draw(fold=-1))
        print("\nMaterialized original (ground truth):\n", qc_ref.draw(fold=-1))
        print("\nRewritten by pass (ZYZ basis):\n", compiled.draw(fold=-1))


def main():
    print("=== Day 6: Diagonality (XZ) — ZYZ rewrite, layout‑aware (Qiskit 1.4) ===")
    run_and_check(*ex1_simple(), show_circuits=True)
    run_and_check(*ex2_sequence(), show_circuits=False)
    run_and_check(*ex3_two_qubit(), show_circuits=False)
    run_and_check(*ex4_multiqubit_mixed_axes(4), show_circuits=False)
    for label, qc, weights in ex5_random_battery(m=8, seed=46):
        run_and_check(label, qc, weights, show_circuits=False)


if __name__ == "__main__":
    main()
