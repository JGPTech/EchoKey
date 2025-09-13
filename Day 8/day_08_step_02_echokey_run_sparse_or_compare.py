# run_sparse_or_compare.py
# Qiskit 1.4 + Aer: run EchoKey QASM on a sparse simulator by default,
# or --compare vs the raw QASM. Prints timing + counts (+ TVD if compare).

from __future__ import annotations
import argparse, time
from collections import Counter
from typing import Dict

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit.circuit.library import RZGate, RYGate
from qiskit_aer import AerSimulator

# -----------------------------
# Robust QASM2 loader (1.4)
# -----------------------------
def load_qasm_file(path: str) -> QuantumCircuit:
    # Try new parser first; fall back to legacy for extended qelib1.inc (sx/sxdg/ch/…)
    from qiskit.qasm2 import loads as q2_loads
    from qiskit.qasm2.exceptions import QASM2ParseError
    with open(path, "r", encoding="utf-8") as f:
        program = f.read()
    try:
        return q2_loads(program)
    except QASM2ParseError:
        return QuantumCircuit.from_qasm_str(program)

# ------------------------------------
# Expand ek_* → native RZ/RY for Aer
# ------------------------------------
def expand_echokey(qc_in: QuantumCircuit) -> QuantumCircuit:
    """Replace ek_frac/ek_rec with native RZ(2θ)/RY(2θ). Keeps any .c_if condition if present."""
    out = QuantumCircuit(*qc_in.qregs, *qc_in.cregs)
    for ci in qc_in.data:
        op: Instruction = ci.operation
        qargs, cargs = ci.qubits, ci.clbits
        # NOTE: Accessing .condition emits a deprecation warning on Qiskit 1.4 (safe to ignore here).
        cond = op.condition
        if op.name == "ek_frac":         # exp(-i θ σz) == RZ(2θ)
            lam2 = 2.0 * float(op.params[0])
            inst = RZGate(lam2)
            if cond is not None:
                inst = inst.c_if(*cond)
            out.append(inst, qargs, cargs)
        elif op.name == "ek_rec":        # exp(-i θ σy) == RY(2θ)
            bet2 = 2.0 * float(op.params[0])
            inst = RYGate(bet2)
            if cond is not None:
                inst = inst.c_if(*cond)
            out.append(inst, qargs, cargs)
        else:
            out.append(op, qargs, cargs)
    return out

# -----------------------------
# Utility: TV distance on counts
# -----------------------------
def l1_distance(counts_a: Dict[str,int], counts_b: Dict[str,int]) -> float:
    total_a = sum(counts_a.values()) or 1
    total_b = sum(counts_b.values()) or 1
    keys = set(counts_a) | set(counts_b)
    return 0.5 * sum(abs(counts_a.get(k,0)/total_a - counts_b.get(k,0)/total_b) for k in keys)

# -----------------------------
# One run (transpile+execute)
# -----------------------------
def run_one(qc: QuantumCircuit, backend: AerSimulator, shots: int, seed: int) -> tuple[Dict[str,int], float, float]:
    backend.set_options(seed_simulator=seed)
    t0 = time.perf_counter()
    tqc = transpile(qc, backend=backend, optimization_level=1)
    t1 = time.perf_counter()
    res = backend.run(tqc, shots=shots).result()
    t2 = time.perf_counter()
    counts = dict(res.get_counts())
    return counts, (t1 - t0), (t2 - t1)  # transpile_s, execute_s

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Run EchoKey QASM sparse-only (default) or --compare vs raw.")
    ap.add_argument("--echokey", required=True, help="EchoKey QASM2 (with opaque ek_* gates)")
    ap.add_argument("--compare", action="store_true", help="Also run the raw QASM and compare counts/timing")
    ap.add_argument("--normal", help="Raw/original QASM2 path (required if --compare)")
    ap.add_argument("--shots", type=int, default=4096, help="Number of shots (default: 4096)")
    ap.add_argument("--seed", type=int, default=123456, help="seed_simulator (default: 123456)")
    ap.add_argument("--sparse_method", default="matrix_product_state",
                    help="Aer method for EchoKey run (default: matrix_product_state)")
    args = ap.parse_args()

    if args.compare and not args.normal:
        ap.error("--compare requires --normal <file>")

    # Load EchoKey QASM and expand ek_* to native gates for simulation
    qc_ek_raw = load_qasm_file(args.echokey)
    qc_ek = expand_echokey(qc_ek_raw)

    # Prepare sparse simulator
    sim_sparse = AerSimulator(method=args.sparse_method)
    if args.sparse_method not in sim_sparse.available_methods():
        print(f"[warn] method='{args.sparse_method}' not in available_methods={sim_sparse.available_methods()}. "
              f"Proceeding; Aer may auto-fallback.")

    # Run sparse-only (default)
    counts_ek, t_trans_ek, t_exec_ek = run_one(qc_ek, sim_sparse, args.shots, args.seed)

    print("\n=== EchoKey (sparse) run ===")
    print(f"Backend: AerSimulator(method={args.sparse_method})")
    print("Timing (seconds): transpile = {:8.4f}   execute = {:8.4f}   total = {:8.4f}"
          .format(t_trans_ek, t_exec_ek, t_trans_ek + t_exec_ek))
    print("Counts (top 10): " + ", ".join(f"{k}:{v}" for k,v in Counter(counts_ek).most_common(10)))

    # Optional comparison path
    if args.compare:
        qc_norm = load_qasm_file(args.normal)
        sim_normal = AerSimulator()  # automatic method selection
        counts_norm, t_trans_norm, t_exec_norm = run_one(qc_norm, sim_normal, args.shots, args.seed)

        same = counts_norm == counts_ek
        dist = l1_distance(counts_norm, counts_ek)

        print("\n=== Comparison vs raw ===")
        print("Backend (normal): AerSimulator(method=automatic)")
        print("Backend (sparse): AerSimulator(method={})".format(args.sparse_method))
        print("\nTiming (seconds):")
        print("  normal: transpile = {:8.4f}   execute = {:8.4f}   total = {:8.4f}"
              .format(t_trans_norm, t_exec_norm, t_trans_norm + t_exec_norm))
        print("  sparse: transpile = {:8.4f}   execute = {:8.4f}   total = {:8.4f}"
              .format(t_trans_ek,   t_exec_ek,   t_trans_ek   + t_exec_ek))
        print("\nExact-match counts?  {}".format("YES ✅" if same else "NO ❌"))
        print("Total-variation distance |p−q|₁/2 = {:.6e}".format(dist))

if __name__ == "__main__":
    main()
