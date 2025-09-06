#!/usr/bin/env python3
"""
8 Days of EchoKey — Day 1+2 Unified Axis Gate (stand‑alone, walkthrough in comments)
CC0‑1.0 — Public Domain

WHAT THIS FILE IS
-----------------
A single, self‑contained Python module that unifies Day‑1 (Cyclicity) and Day‑2 (Recursion)
into a generic one‑qubit EchoKey gate and a layout‑aware Qiskit 1.4 rewrite pass.

    EchoKeyAxisGate(k, theta)  ≡  exp(-i * theta * (a_k · σ))      for k ∈ {0..6}

It rewrites the symbolic EchoKey gate to a native Z–Y–Z Euler factorization using the exact
axis–angle unitary. The per‑wire axis a_k is resolved from a local 7×3 frame A^(p) (one per
physical qubit p), so the pass is layout‑aware.

The bottom of the file includes a runnable test‑bench:
  • Day‑1 demo (k=0), Day‑2 demo (k=1)
  • SU(2) span demo using two non‑colinear axes (here X and Y)
  • Lie‑algebra commutator check: [a·σ, b·σ] = 2i (a×b)·σ
  • Per‑site frames + routing‑agnostic fidelity checks

MATH WALKTHROUGH (concise)
--------------------------
1) EchoKey local generators:  E_k^∘ := a_k · σ  with  a_k ∈ ℝ^3,  ||a_k|| = 1.  (Rows of A.)
2) Gate definition:           U_k(θ) := exp(-i θ (a_k · σ)).
3) Axis–angle form:           U(φ, n̂) = cos(φ/2) I − i sin(φ/2) (n̂ · σ),  with  φ = 2θ, n̂ = a_k.
4) ZYZ completeness:          ∀U∈SU(2), ∃ α,β,γ s.t. U ≍ RZ(α) RY(β) RZ(γ) (≍ up to global phase).
5) Rewrite rule:              ek_axis(k,θ)  →  RZ(α) RY(β) RZ(γ),  where (α,β,γ)=ZYZ(U(2θ, a_k)).
6) Layout awareness:          If logical qubit q maps to physical p, use A^(p) to read a_k.
7) SU(2) span (two axes):     If a and b are not colinear, then {a·σ, b·σ, (a×b)·σ} spans su(2),
                              since [a·σ, b·σ] = 2i (a×b)·σ. Hence two non‑colinear axes generate
                              the full single‑qubit algebra; compositions achieve arbitrary SU(2).

IMPLEMENTATION NOTES
--------------------
• We synthesize exactly from the 2×2 matrix — no approximations; fidelities hit ~1.0 numerically.
• We keep everything small/explicit; no reliance on transpiler stages beyond a PassManager that
  runs only our rewrite pass (for apples‑to‑apples unitary checks).

Run me:
  $ python echokey_axisgate.py

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import expm

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.library import RZGate, RYGate, HGate, RXGate, CXGate, UnitaryGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer  # Qiskit 1.4
from qiskit.transpiler import PassManager

# =============================================================================
# 0) Local frames A and helpers
# =============================================================================

def ek7_A_default() -> np.ndarray:
    """Return a baseline 7×3 coefficient matrix A whose rows are unit vectors a_k^T.
    Row indices: 0→cyclicity (X), 1→recursion (Y), 2→Z, others are simple diagonals.
    """
    A = np.array([
        [ 1, 0, 0],  # k=0 : X (Day‑1 canonical pick)
        [ 0, 1, 0],  # k=1 : Y (Day‑2 canonical pick)
        [ 0, 0, 1],  # k=2 : Z
        [ 1, 1, 0],
        [ 0, 1, 1],
        [ 1, 0, 1],
        [ 1, 1, 1],
    ], dtype=float)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A


def axis_from_angles(theta_xy: float, tilt_z: float) -> np.ndarray:
    """Build a unit vector n̂ from a planar angle and a Z‑tilt."""
    v = np.array([np.cos(theta_xy), np.sin(theta_xy), tilt_z], float)
    n = np.linalg.norm(v)
    return v / (n if n > 0 else 1.0)


def site_specific_A(num_qubits: int) -> Dict[int, np.ndarray]:
    """Attach a distinct A^(p) to each physical wire p. Day‑2 used a per‑site Y‑axis; we generalize.
    For variety, rotate the k=1 row around the XY‑circle while keeping others as defaults.
    """
    As: Dict[int, np.ndarray] = {}
    base = ek7_A_default()
    for p in range(num_qubits):
        A = base.copy()
        a2 = axis_from_angles(theta_xy=2*np.pi*p/max(1, num_qubits), tilt_z=0.25*np.sin(p + 0.25))
        A[1] = a2 / np.linalg.norm(a2)
        As[p] = A
    return As

# =============================================================================
# 1) Symbolic EchoKey gate for any k ∈ {0..6}
# =============================================================================

class EchoKeyAxisGate(Gate):
    """Symbolic EchoKey instruction: ek_axis(k, θ) ≡ exp(-i θ (a_k · σ)).
    We store k as an attribute; θ is the sole Qiskit parameter (float).
    """
    def __init__(self, k: int, theta: float):
        if not (0 <= int(k) <= 6):
            raise ValueError("k must be in 0..6")
        self._k = int(k)
        super().__init__(name="ek_axis", num_qubits=1, params=[float(theta)])
        # Symbolic placeholder; the rewrite pass will substitute a native circuit.
        self.definition = QuantumCircuit(1, name=f"ek_axis[{self._k}]")

    @property
    def k(self) -> int:
        return self._k

# =============================================================================
# 2) Axis–angle → ZYZ helper (exact angles from exact 2×2 unitary)
# =============================================================================

_DECOMP_ZYZ = OneQubitEulerDecomposer(basis="ZYZ")

def axis_angle_to_unitary(nhat: np.ndarray, phi: float) -> np.ndarray:
    """Return U = cos(φ/2) I − i sin(φ/2) (n̂·σ). nhat must be unit length."""
    nx, ny, nz = nhat
    I2 = np.eye(2, dtype=complex)
    SX = np.array([[0, 1],[1, 0]], complex)
    SY = np.array([[0,-1j],[1j, 0]], complex)
    SZ = np.array([[1, 0],[0,-1 ]], complex)
    return np.cos(phi/2)*I2 - 1j*np.sin(phi/2)*(nx*SX + ny*SY + nz*SZ)

# =============================================================================
# 3) Layout‑aware rewrite pass: ek_axis(k,θ) → RZ · RY · RZ
# =============================================================================

@dataclass
class SiteWeights:
    A: np.ndarray  # 7×3 per physical wire

class EchoKeyAxisRewritePass(TransformationPass):
    """Replace each EchoKeyAxisGate with an exactly synthesized ZYZ circuit using the
    per‑wire axis a_k from weights_per_qubit[p].A[k]. Layout from property_set if present.
    """
    def __init__(self, weights_per_qubit: Dict[int, SiteWeights]):
        super().__init__()
        self.W = weights_per_qubit
        self._decomp = OneQubitEulerDecomposer(basis="ZYZ")

    # Resolve physical index for a 1q node under the current layout (if any)
    def _phys_index(self, dag: DAGCircuit, node) -> int:
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
            op = node.op
            if isinstance(op, EchoKeyAxisGate):
                theta = float(op.params[0])
                k = int(op.k)
                p = self._phys_index(dag, node)
                A = self.W.get(p, SiteWeights(ek7_A_default())).A
                if not (0 <= k < A.shape[0]):
                    raise ValueError(f"k={k} out of range for A")
                ak = A[k]
                if not np.isfinite(ak).all() or np.linalg.norm(ak) < 1e-15:
                    raise ValueError(f"A[{k}] invalid/degenerate for qubit {p}")
                nhat = ak / np.linalg.norm(ak)

                # Exact SU(2) for ek_axis(k,θ): U = cos(θ) I − i sin(θ) (n̂·σ)
                U = np.cos(theta)*I2 - 1j*np.sin(theta)*(nhat[0]*SX + nhat[1]*SY + nhat[2]*SZ)

                # Decompose to ZYZ and substitute
                rep_qc = self._decomp(U)  # 1‑qubit circuit on {RZ, RY}
                dag.substitute_node_with_dag(node, circuit_to_dag(rep_qc))
        return dag

# =============================================================================
# 4) Utilities: unitary, fidelity, materialization, compile‑with‑pass‑only
# =============================================================================

def unitary(circ: QuantumCircuit) -> np.ndarray:
    return Operator(circ).data


def fid(U: np.ndarray, V: np.ndarray) -> float:
    d = U.shape[0]
    return float(abs(np.trace(U.conj().T @ V)) / d)


def materialize_ek_ops(qc: QuantumCircuit, weights: Dict[int, SiteWeights]) -> QuantumCircuit:
    """Replace each EchoKeyAxisGate with its exact 2×2 unitary (ground truth)."""
    out = QuantumCircuit(qc.num_qubits)
    for creg in qc.cregs:
        out.add_register(creg)

    SX = np.array([[0, 1],[1, 0]], complex)
    SY = np.array([[0,-1j],[1j, 0]], complex)
    SZ = np.array([[1, 0],[0,-1 ]], complex)

    for inst in qc.data:  # CircuitInstruction (Qiskit 1.4)
        op, qubits, clbits = inst.operation, inst.qubits, inst.clbits
        if isinstance(op, EchoKeyAxisGate):
            theta = float(op.params[0]); k = int(op.k)
            q_index = qc.find_bit(qubits[0]).index  # logical index in reference path
            A = weights.get(q_index, SiteWeights(ek7_A_default())).A
            ak = A[k] / np.linalg.norm(A[k])
            H = theta * (ak[0]*SX + ak[1]*SY + ak[2]*SZ)
            U = expm(-1j * H)
            out.append(UnitaryGate(U, label=f"ek_axis[{k}](mat)"), [qubits[0]])
        else:
            out.append(op, qubits, clbits)
    return out


def compile_with_pass_only(qc: QuantumCircuit, weights: Dict[int, SiteWeights]) -> QuantumCircuit:
    pm = PassManager([EchoKeyAxisRewritePass(weights)])
    return pm.run(qc)

# Axis–angle recovery from a 2×2 SU(2) unitary U = cos(φ/2) I − i sin(φ/2) (n̂·σ)
# Compute φ from Tr(U) and n̂ from Pauli projections.
from typing import Tuple

def unitary_to_axis_angle(U: np.ndarray) -> Tuple[np.ndarray, float]:
    trU = np.trace(U)
    c = float(np.clip(np.real(trU) / 2.0, -1.0, 1.0))
    phi = 2.0 * float(np.arccos(c))
    if abs(np.sin(phi / 2.0)) < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    SX = np.array([[0, 1],[1, 0]], complex)
    SY = np.array([[0,-1j],[1j, 0]], complex)
    SZ = np.array([[1, 0],[0,-1 ]], complex)
    s = 2.0 * np.sin(phi / 2.0)
    nx = float(np.imag(np.trace(SX @ U)) / s)
    ny = float(np.imag(np.trace(SY @ U)) / s)
    nz = float(np.imag(np.trace(SZ @ U)) / s)
    n = np.array([nx, ny, nz], float)
    n /= np.linalg.norm(n) if np.linalg.norm(n) > 0 else 1.0
    return n, phi

# =============================================================================
# 5) Demos / Examples
# =============================================================================

def ex_two_gate_pair() -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    """Two‑gate demo using ek_cyc (k=0) then ek_rec (k=1).
    This does NOT span all SU(2) by itself (2 parameters vs 3), but together with commutators
    it generates su(2). We print the effective axis‑angle of the 2‑gate product.
    """
    qc = QuantumCircuit(1)
    qc.append(EchoKeyAxisGate(0, 0.6), [0])   # ek_cyc ~ X‑axis
    qc.append(EchoKeyAxisGate(1, -0.8), [0])  # ek_rec ~ Y‑axis
    weights = {0: SiteWeights(A=ek7_A_default())}
    return ("[2‑gate] ek_cyc(0.6) · ek_rec(-0.8)", qc, weights)

# (rest of Demos follow)
# =============================================================================

def ex_day1_demo() -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    """Day‑1 (Cyclicity): use k=0 (X‑axis) once, then H."""
    qc = QuantumCircuit(1)
    qc.append(EchoKeyAxisGate(0, 0.37), [0])
    qc.append(HGate(), [0])
    weights = {0: SiteWeights(A=ek7_A_default())}
    return ("[Day‑1] 1q ek_axis(k=0) → H", qc, weights)


def ex_day2_demo() -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    """Day‑2 (Recursion): use k=1 (Y‑axis) twice with some native rotations."""
    qc = QuantumCircuit(1)
    qc.append(RZGate(0.15), [0])
    qc.append(EchoKeyAxisGate(1, -0.42), [0])
    qc.append(RYGate(0.20), [0])
    qc.append(EchoKeyAxisGate(1, 0.81), [0])
    qc.append(RXGate(-0.31), [0])
    weights = {0: SiteWeights(A=ek7_A_default())}
    return ("[Day‑2] 1q sequence with ek_axis(k=1)", qc, weights)


def ex_su2_span_two_axes() -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    """SU(2) span using two non‑colinear axes: k=0 (X) and k=1 (Y).
    A classic ZYZ factorization uses Z and Y; XYX also spans SU(2). Here we just show that a
    short composition ek[X](α) ek[Y](β) ek[X](γ) rewrites exactly to ZYZ (thus arbitrary SU(2)).
    """
    qc = QuantumCircuit(1)
    qc.append(EchoKeyAxisGate(0,  0.7), [0])
    qc.append(EchoKeyAxisGate(1, -0.9), [0])
    qc.append(EchoKeyAxisGate(0,  0.3), [0])
    weights = {0: SiteWeights(A=ek7_A_default())}
    return ("[SU2] 1q XYX composition (k=0,1,0)", qc, weights)


def ex_commutator_check() -> None:
    """Numerically verify [a·σ, b·σ] = 2i (a×b)·σ for a=a0 (X), b=a1 (Y)."""
    A = ek7_A_default(); a = A[0]; b = A[1]
    SX = np.array([[0, 1],[1, 0]], complex)
    SY = np.array([[0,-1j],[1j, 0]], complex)
    SZ = np.array([[1, 0],[0,-1 ]], complex)
    def m_from(v):
        return v[0]*SX + v[1]*SY + v[2]*SZ
    LHS = m_from(a) @ m_from(b) - m_from(b) @ m_from(a)
    RHS = 2j * m_from(np.cross(a, b))
    err = np.linalg.norm(LHS - RHS)
    print(f"[Lie] ||[a·σ,b·σ] - 2i (a×b)·σ||_F = {err:.3e}")


def ex_anticommutator_check() -> None:
    """Jordan product sanity: {a·σ, b·σ} = 2 (a·b) I."""
    A = ek7_A_default(); a = A[0]; b = A[1]
    SX = np.array([[0, 1],[1, 0]], complex)
    SY = np.array([[0,-1j],[1j, 0]], complex)
    SZ = np.array([[1, 0],[0,-1 ]], complex)
    I2 = np.eye(2, dtype=complex)
    def m_from(v):
        return v[0]*SX + v[1]*SY + v[2]*SZ
    LHS = m_from(a) @ m_from(b) + m_from(b) @ m_from(a)
    RHS = 2.0 * float(np.dot(a, b)) * I2
    err = np.linalg.norm(LHS - RHS)
    print(f"[Jordan] ||{a}·σ ∘ {b}·σ − 2(a·b)I||_F = {err:.3e}")


def ex_multiqubit_per_site_axes(n: int = 4) -> Tuple[str, QuantumCircuit, Dict[int, SiteWeights]]:
    qc = QuantumCircuit(n)
    As = site_specific_A(n)
    # Place mixed axes on different wires, with entanglers between them
    for q in range(n):
        k = 0 if (q % 2 == 0) else 1
        qc.append(EchoKeyAxisGate(k, 0.17*(q+1)), [q])
        if q < n-1:
            qc.append(CXGate(), [q, q+1])
    weights = {q: SiteWeights(A=As[q]) for q in range(n)}
    return (f"[Multi] {n}q per‑site A^(p) + CX chain", qc, weights)

# =============================================================================
# 6) Runner with fidelity checks
# =============================================================================

def run_and_check(label: str, qc: QuantumCircuit, weights: Dict[int, SiteWeights], show_circuits=False):
    compiled = compile_with_pass_only(qc, weights)
    qc_ref = materialize_ek_ops(qc, weights)
    U_in, U_out = unitary(qc_ref), unitary(compiled)
    F = fid(U_in, U_out)
    print(f"{label:36s}  fidelity = {F:.12f}")
    if show_circuits:
        print("\nOriginal (symbolic):\n", qc.draw(fold=-1))
        print("\nMaterialized original (ground truth):\n", qc_ref.draw(fold=-1))
        print("\nRewritten by pass (ZYZ basis):\n", compiled.draw(fold=-1))


def main():
    print("=== Day 1+2: Unified Axis Gate → ZYZ (Qiskit 1.4) — layout‑aware ===")

    # 2‑gate pair (ek_cyc then ek_rec) — show effective axis‑angle and fidelity
    label2, qc2, w2 = ex_two_gate_pair()
    run_and_check(label2, qc2, w2, show_circuits=False)
    U2 = unitary(compile_with_pass_only(qc2, w2))
    n2, phi2 = unitary_to_axis_angle(U2)
    print(f"{label2}  ⇒  axis ≈ {n2},  angle φ ≈ {phi2:.6f}")

    # Day‑1 and Day‑2 quick demos
    run_and_check(*ex_day1_demo(), show_circuits=True)
    run_and_check(*ex_day2_demo(), show_circuits=False)

    # SU(2) span via two non‑colinear axes (XYX composition)
    run_and_check(*ex_su2_span_two_axes(), show_circuits=False)

    # Lie‑algebra commutator (should be ~0)
    ex_commutator_check()

    # Jordan (anti‑commutator) closure (should be ~0)
    ex_anticommutator_check()

    # Multi‑qubit per‑site frames
    run_and_check(*ex_multiqubit_per_site_axes(4), show_circuits=False)


if __name__ == "__main__":
    main()

