#!/usr/bin/env python3
# echokey_toy_model.py
"""
EchoKey 7-Operator Toy → Emergent Pauli & Circuit Equivalence
-------------------------------------------------------------
This is a compact, reproducible toy you can point to for the “8 Days of EchoKey”.

What it does (default run):
  1) Build a small hex-patch of sites (37 nodes by default) for indexing only.
  2) At each site i, define 7 local EchoKey operators E_k^(i) (2x2 Hermitian).
  3) Constructively recover {σx,σy,σz} from {E_k} via a right-inverse:
       B^(i) ∈ R^{3×7}:  σ_a^(i) = Σ_k B^(i)[a,k] · E_k^(i)∘   (trace-removed)
     → Prints per-site orthonormality and commutator residuals.
  4) Single-qubit circuit check: random sequence of exp(-i θ Σ_k c_k E_k) vs
     exp(-i θ Σ_a α_a σ_a) where α = B · c  ⇒ unitary equivalence.
  5) Two-qubit check on one edge: EchoKey-XYZ vs Pauli-XYZ (compiled via B).
  6) Export per-site injection weights to JSON: echokey_pauli_injection.json

Use this as the reference design for UCC passes:
  - Day 1–7: ship each EchoKey operator (plus helpers).
  - Day 8: ship the emergence proof + rewrite (this file’s Step 3–5).

Deps: numpy, scipy, networkx
"""

import json
import argparse
from dataclasses import dataclass
import numpy as np
import networkx as nx
from scipy.linalg import expm

# -------------------------
# Pauli basis & primitives
# -------------------------
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = np.stack([SX, SY, SZ], axis=0)  # [3,2,2]

def pauli_decompose(H: np.ndarray):
    """Return (a0, a) s.t. H = a0*I + a·σ   with a∈R^3."""
    a0 = 0.5 * np.trace(H).real
    a = np.array([0.5*np.trace(H @ P).real for P in PAULI], float)
    return a0, a

def pauli_recompose(a0: float, a3: np.ndarray):
    return a0*I2 + a3[0]*SX + a3[1]*SY + a3[2]*SZ

def frob(A: np.ndarray) -> float:
    return float(np.sqrt(np.real(np.trace(A.conj().T @ A))))

# -------------------------
# Hex patch (for indexing)
# -------------------------
def hex_patch_radius(R=3):
    """
    Return (G, axial) where:
      G  : undirected graph on ~3R(R+1)+1 nodes
      axial: dict idx -> (q,r)
    """
    axial_nodes = []
    for q in range(-R, R + 1):
        rmin = max(-R, -q - R)
        rmax = min(R, -q + R)
        for r in range(rmin, rmax + 1):
            axial_nodes.append((q, r))
    idx_of = {ax: i for i, ax in enumerate(axial_nodes)}
    AX_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    G = nx.Graph()
    G.add_nodes_from(range(len(axial_nodes)))
    for i, (q, r) in enumerate(axial_nodes):
        for dq, dr in AX_DIRS:
            nb = (q + dq, r + dr)
            if nb in idx_of:
                G.add_edge(i, idx_of[nb])
    axial = {i: axial_nodes[i] for i in range(len(axial_nodes))}
    return G, axial

def axial_to_cube(qr):
    q, r = qr
    return np.array([q, r, -q - r], int)  # (x,y,z) with sum 0

# ------------------------------------
# EchoKey 7 operators (single-qubit)
# ------------------------------------
@dataclass
class EK7Params:
    """Knobs to shape the 7 operator directions & scaling."""
    id_leak: float = 0.05  # small identity leakage (later removed)
    norm_each: bool = True # normalize direction rows

def ek7_directions_from_cube(cube_xyz: np.ndarray) -> np.ndarray:
    """
    Build an interpretable 7x3 matrix of coefficient vectors (rows)
    using the cube coords (x,y,z) to fix a local orientation:
      rows:  ex, ey, ez, ex+ey, ey+ez, ex+ez, ex+ey+ez (signed)
    """
    x, y, z = cube_xyz
    ex = np.array([np.sign(x) or 1.0, 0.0, 0.0], float)
    ey = np.array([0.0, np.sign(y) or 1.0, 0.0], float)
    ez = np.array([0.0, 0.0, np.sign(z) or 1.0], float)
    A = np.stack([
        ex,
        ey,
        ez,
        ex + ey,
        ey + ez,
        ex + ez,
        ex + ey + ez,
    ], axis=0)
    # Normalize rows to unit length (scale-agnostic)
    A /= np.maximum(1e-12, np.linalg.norm(A, axis=1, keepdims=True))
    return A  # shape (7,3)

def make_echokey7_ops(A7: np.ndarray, p: EK7Params) -> list:
    """
    Given A7 (7x3) direction rows, return 7 Hermitian 2x2 ops.
    Each E_k = a0_k*I + a_k · σ  (then later we remove trace).
    """
    rng = np.random.default_rng(7)
    E = []
    for k in range(7):
        a0 = p.id_leak * rng.uniform(-1, 1)   # tiny identity leakage
        ak = A7[k] / (np.linalg.norm(A7[k]) if p.norm_each else 1.0)
        H = pauli_recompose(a0, ak)
        E.append(0.5*(H + H.conj().T))
    return E

# -----------------------------------
# Emergence: 7 → {σx,σy,σz} per site
# -----------------------------------
def build_A_from_ops(E_list):
    """Return (A 7x3, E0 list traceless)."""
    A = np.zeros((7, 3), float)
    E0 = []
    for H in E_list:
        a0, a = pauli_decompose(H)
        H0 = H - a0*I2
        A[k:=len(E0)] = pauli_decompose(H0)[1]
        E0.append(H0)
    return A, E0

def right_inverse(A: np.ndarray, reg: float = 0.0):
    """B = (AᵀA + reg I)^{-1} Aᵀ  (shape 3x7)."""
    G = A.T @ A
    if reg > 0:
        G = G + reg*np.eye(3)
    return np.linalg.solve(G, A.T)

def emergence_report(E_list, verbose=False):
    """Compute B, σ' = Σ_k B E_k, and diagnostics."""
    A, E0 = build_A_from_ops(E_list)
    s = np.linalg.svd(A, compute_uv=False)
    rank = int((s > 1e-10).sum())
    cond = (s[0]/s[-1]) if s[-1] > 0 else np.inf
    lam = 0.0 if np.isfinite(cond) and cond < 1e6 else 1e-8
    B = right_inverse(A, reg=lam)  # 3x7
    # Construct S_i
    S = []
    for i in range(3):
        Si = sum(B[i,k]*E0[k] for k in range(7))
        S.append(0.5*(Si + Si.conj().T))
    # Orthonormality
    O = np.array([[0.5*np.trace(S[a]@S[b]).real for b in range(3)] for a in range(3)])
    # Commutator residuals
    eps = np.zeros((3,3,3), int)
    eps[0,1,2]=eps[1,2,0]=eps[2,0,1]= 1
    eps[2,1,0]=eps[0,2,1]=eps[1,0,2]=-1
    resids = []
    for a in range(3):
        for b in range(3):
            comm = S[a]@S[b] - S[b]@S[a]
            target = 2j*sum(eps[a,b,c]*S[c] for c in range(3))
            resids.append(frob(comm - target))
    resids = np.array(resids)
    if verbose:
        print("  s(A)=", np.round(s, 6), " rank=", rank, " cond≈", f"{cond:.3e}")
        print("  Orthonormality:\n", np.round(O, 6))
        print("  Commutator residuals: mean=", f"{resids.mean():.3e}", " max=", f"{resids.max():.3e}")
    return {"A":A, "B":B, "S":S, "O":O,
            "resid_mean":float(resids.mean()),
            "resid_max":float(resids.max()),
            "rank":int(rank), "cond":float(cond)}

# --------------------------------------
# Circuit checks (single & two qubits)
# --------------------------------------
def ek_single_qubit_unitary(E7, coeffs7, theta):
    """U_EK = exp(-i * theta * sum_k coeffs7[k] * E_k^∘)."""
    E0 = []
    for H in E7:
        a0, a = pauli_decompose(H)
        E0.append(H - a0*I2)
    H_eff = sum(coeffs7[k]*E0[k] for k in range(7))
    return expm(-1j * theta * H_eff)

def pauli_single_qubit_unitary(alpha3, theta):
    """U_P = exp(-i * theta * (alpha·σ))."""
    H = pauli_recompose(0.0, alpha3)
    return expm(-1j * theta * H)

def unitary_fidelity(U, V):
    """F = |Tr(U† V)| / 2 for 2x2 unitaries (global phase invariant via |Tr| metric)."""
    return abs(np.trace(U.conj().T @ V)) / U.shape[0]

def ek_two_qubit_xyz(E7_i, E7_j, B_i, B_j, J=(1.0,0.7,0.4), t=0.35):
    """
    Build EchoKey-XYZ Hamiltonian using site i,j:
      H_EK = Jx Sx_i Sx_j + Jy Sy_i Sy_j + Jz Sz_i Sz_j,
    where S_• = Σ_k B[•,k] E_k^∘ (constructed from E7 at each site).
    Compare exp(-i t H_EK) with Pauli XYZ unitary.
    """
    # Construct S at i, j
    def S_from(E7, B):
        E0 = []
        for H in E7:
            E0.append(H - pauli_decompose(H)[0]*I2)
        S = []
        for a in range(3):
            Sa = sum(B[a,k]*E0[k] for k in range(7))
            S.append(0.5*(Sa + Sa.conj().T))
        return S
    Si = S_from(E7_i, B_i); Sj = S_from(E7_j, B_j)
    Sxi,Syi,Szi = Si; Sxj,Syj,Szj = Sj
    Jx,Jy,Jz = J
    H_EK = (Jx*np.kron(Sxi,Sxj) + Jy*np.kron(Syi,Syj) + Jz*np.kron(Szi,Szj))
    U_EK = expm(-1j * t * H_EK)
    # Pauli ref
    H_P = (Jx*np.kron(SX,SX) + Jy*np.kron(SY,SY) + Jz*np.kron(SZ,SZ))
    U_P = expm(-1j * t * H_P)
    return U_EK, U_P

# -------------------
# Orchestrator
# -------------------
def main():
    ap = argparse.ArgumentParser(description="EchoKey 7-Operator Toy → Emergent Pauli")
    ap.add_argument("--radius", type=int, default=3, help="hex patch radius (3→37 nodes)")
    ap.add_argument("--single_circ_len", type=int, default=6, help="random single-qubit op count")
    ap.add_argument("--seed", type=int, default=123, help="rng seed")
    ap.add_argument("--export_json", action="store_true", help="write echokey_pauli_injection.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # (1) Hex patch (indexing only)
    G, axial = hex_patch_radius(args.radius)
    cubes = {i: axial_to_cube(axial[i]) for i in G.nodes()}
    print(f"[1] Hex patch: R={args.radius}  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

    # (2) Build EK7 per site
    params = EK7Params()
    E7_of = {}
    for i in G.nodes():
        A7 = ek7_directions_from_cube(cubes[i])
        E7_of[i] = make_echokey7_ops(A7, params)
    print("[2] EchoKey 7 local operators instantiated at all sites.")

    # (3) Emergence per site → Pauli, with diagnostics
    reports = {}
    pass_ct = 0
    for i in G.nodes():
        rep = emergence_report(E7_of[i], verbose=False)
        reports[i] = rep
        ok = (rep["rank"]==3 and np.allclose(rep["O"], np.eye(3), atol=1e-12, rtol=0) and rep["resid_max"]<1e-12)
        pass_ct += int(ok)
    print(f"[3] Emergent Pauli per-site: {pass_ct}/{G.number_of_nodes()} sites PASS (rank=3, ortho≈I, comm residual ≤1e-12).")
    worst = max(G.nodes(), key=lambda i: reports[i]["resid_max"])
    print(f"    Worst commutator residual at site {worst}: {reports[worst]['resid_max']:.3e}")

    # (4) Single-qubit random circuit equivalence at a random site
    site = int(rng.choice(list(G.nodes())))
    A = reports[site]["A"]  # 7x3
    # Random steps: coeffs7 (normalized) and angle
    U_EK = np.eye(2, dtype=complex)
    U_P  = np.eye(2, dtype=complex)
    for _ in range(args.single_circ_len):
        c7 = rng.normal(size=7); c7 /= np.linalg.norm(c7)
        alpha = A.T @ c7  # R^3  (this is the exact forward map)
        theta = 0.2 * rng.uniform()  # small-ish rotation
        U_EK = ek_single_qubit_unitary(E7_of[site], c7, theta) @ U_EK
        U_P  = pauli_single_qubit_unitary(alpha, theta) @ U_P
    F = unitary_fidelity(U_EK, U_P)
    print(f"[4] Single-qubit circuit @ site {site}: fidelity |Tr(U†V)|/2 = {F:.12f}")

    # (5) Two-qubit EchoKey-XYZ vs Pauli-XYZ on one edge
    #     pick a random edge (i,j)
    i, j = list(G.edges())[rng.integers(0, len(G.edges()))]
    Uek, Up = ek_two_qubit_xyz(E7_of[i], E7_of[j], reports[i]["B"], reports[j]["B"], J=(1.0,0.7,0.4), t=0.35)
    # Use average gate fidelity proxy via normalized trace (4x4 → divide by 4)
    F2 = abs(np.trace(Uek.conj().T @ Up)) / 4.0
    print(f"[5] Two-qubit XYZ on edge ({i},{j}): fidelity = {F2:.12f}")

    # (6) Export per-site B matrices (injection weights)
    if args.export_json:
        out = {int(i): reports[i]["B"].tolist() for i in G.nodes()}
        with open("echokey_pauli_injection.json", "w") as f:
            json.dump({"schema":"ek7→pauli-weights-v1","B_per_site":out}, f, indent=2)
        print('[6] Wrote echokey_pauli_injection.json (3×7 weights per site).')

if __name__ == "__main__":
    main()
