#!/usr/bin/env python3
"""
8 Days of EchoKey — Day 8: QASM → EchoKey QASM converter (exact, layout‑agnostic)
CC0‑1.0 — Public Domain

GOAL
----
Given an OpenQASM 2.0 circuit, fuse every per‑qubit chain of 1‑qubit gates into at most
**three EchoKey gates** using the identity

    RZ(α) · RY(β) · RZ(γ)  ≡  exp(−i (α/2) σ_z) · exp(−i (β/2) σ_y) · exp(−i (γ/2) σ_z)
                          ≡  ek_frac(α/2) · ek_rec(β/2) · ek_frac(γ/2)

and emit a new **QASM 2.0** program that contains only standard multi‑qubit ops (cx, cz, …),
barriers, measurement/reset, plus two **opaque EchoKey gates** `ek_rec` (Y axis) and
`ek_frac` (Z axis). The transformation is **exact** up to a global phase; final bitstrings
are preserved 1‑to‑1.

WHY IT'S CHEAPER
----------------
Long runs of 1‑qubit gates become at most 3 EchoKey operations on each wire between
entanglers/measurements/barriers/conditionals. Back‑ends that natively implement the
EchoKey family (or compile them efficiently) will execute fewer operations.

USAGE
-----
  $ python day_08_qasm_to_echokey.py --in path/to/input.qasm --out path/to/output_ek.qasm

Optional:
  --print  : print the converted QASM to stdout
  --verify : numerically check unitary equivalence (skips past the first measurement/reset)

NOTES
-----
• This converter is **layout‑agnostic**: it uses canonical Y and Z axes for ek_rec/ek_frac.
  (Your downstream layout‑aware EchoKey pass can still rewrite them as needed.)
• We do **no approximations**. Angles are computed from the exact 2×2 unitary of each fused
  chain using Qiskit's ZYZ decomposer.
• We DO respect barriers, entanglers (≥2‑qubit ops), conditionals, resets, and measurements:
  those boundaries trigger a flush of any pending 1‑qubit fusion on the touched wires.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer  # Qiskit 1.4

# -------------------------
# Helpers: formatting
# -------------------------

def fmt_angle(x: float) -> str:
    """Compact, QASM‑safe float formatting."""
    return f"{float(x):.16g}"

# -------------------------
# Core: fuse per‑wire 1q chains
# -------------------------

ACCUMULATABLE_1Q = {
    # Standard library 1q ops
    "u", "u1", "u2", "u3", "p", "rx", "ry", "rz", "id",
    "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg",
}

BOUNDARY_OPS = {"barrier", "measure", "reset"}

@dataclass
class Accum:
    U: np.ndarray  # accumulated 2×2 unitary (latest on the left)
    has: bool


def is_boundary(inst: Instruction) -> bool:
    return (inst.name in BOUNDARY_OPS) or (inst.num_qubits != 1) or (inst.condition is not None)


def is_1q_unitary(inst: Instruction) -> bool:
    return (inst.num_qubits == 1) and (inst.name in ACCUMULATABLE_1Q) and (inst.condition is None)


def op_unitary(inst: Instruction) -> np.ndarray:
    """Exact 2×2 matrix for a 1‑qubit instruction (numbers only)."""
    qc = QuantumCircuit(1)
    qc.append(inst, [0])
    return Operator(qc).data

# -------------------------
# Convert: QC → EchoKey‑QASM
# -------------------------

class QasmEchoKeyEmitter:
    def __init__(self, qc: QuantumCircuit):
        self.qc = qc
        self.decomp = OneQubitEulerDecomposer(basis="ZYZ")

        # Map flat qubit index -> "qreg[i]" token for QASM emission
        self.qbit_tok: List[str] = []
        for qreg in qc.qregs:
            for i in range(qreg.size):
                self.qbit_tok.append(f"{qreg.name}[{i}]")

        # (Optional) helper for classical bit tokens, used only when emitting measure/reset
        def _ctok(cb) -> str:
            reg = getattr(cb, "register", getattr(cb, "_register", None))
            idx = getattr(cb, "index", getattr(cb, "_index", None))
            return f"{reg.name}[{idx}]"
        self._ctok = _ctok

        # One accumulator per qubit
        self.acc: Dict[int, Accum] = {i: Accum(U=np.eye(2, dtype=complex), has=False)
                                      for i in range(qc.num_qubits)}
        self.lines: List[str] = []
        self.first_measure_index: Optional[int] = None

    # Small helpers
    @staticmethod
    def _qidx(qa) -> int:
        # robust against API changes
        return getattr(qa, "index", getattr(qa, "_index"))

    def _qtok(self, qa) -> str:
        return self.qbit_tok[self._qidx(qa)]

    # --- flushing logic ---
    def _flush_qubit(self, qidx: int):
        acc = self.acc[qidx]
        if not acc.has:
            return

        # Decompose exact 2×2 into ZYZ on a throwaway circuit
        U = acc.U
        rep = self.decomp(U)  # 1-qubit circuit of RZ, RY, RZ

        # Emit ek_* on the ORIGINAL target qubit index (qidx),
        # do NOT read qargs from the decomposer circuit.
        target = self.qbit_tok[qidx]
        for ci in rep.data:
            op = ci.operation
            if op.name == "rz":
                lam = float(op.params[0])
                self.lines.append(f"ek_frac({fmt_angle(lam/2)}) {target};")
            elif op.name == "ry":
                beta = float(op.params[0])
                self.lines.append(f"ek_rec({fmt_angle(beta/2)}) {target};")
            else:
                raise RuntimeError(f"Unexpected gate in ZYZ rep: {op.name}")

        # reset accumulator
        self.acc[qidx] = Accum(U=np.eye(2, dtype=complex), has=False)

    def _flush_all(self):
        for q in range(self.qc.num_qubits):
            self._flush_qubit(q)

    # --- emission helpers for non-fused ops ---
    def _emit_instruction(self, inst: Instruction, qargs, cargs):
        """Emit a standard QASM line for inst (non-EchoKey)."""
        # Optional classical condition
        prefix = ""
        if inst.condition is not None:
            creg, val = inst.condition
            prefix = f"if ({creg.name}=={val}) "

        name = inst.name

        if name in {"cx", "cz", "swap", "ch", "ccx", "cswap"}:
            qs = ", ".join(self._qtok(qa) for qa in qargs)
            self.lines.append(f"{prefix}{name} {qs};")

        elif name == "barrier":
            qs = ", ".join(self._qtok(qa) for qa in qargs)
            self.lines.append(f"barrier {qs};")

        elif name == "measure":
            qa = qargs[0]; ca = cargs[0]
            self.lines.append(f"measure {self._qtok(qa)} -> {self._ctok(ca)};")
            if self.first_measure_index is None:
                self.first_measure_index = len(self.lines) - 1

        elif name == "reset":
            qa = qargs[0]
            self.lines.append(f"reset {self._qtok(qa)};")

        else:
            # Fallback (rare): print the original line as-is
            qs = ", ".join(self._qtok(qa) for qa in qargs)
            params = inst.params
            if params:
                par = ", ".join(fmt_angle(float(p)) for p in params)
                self.lines.append(f"{prefix}{name}({par}) {qs};")
            else:
                self.lines.append(f"{prefix}{name} {qs};")

    # --- main walk ---
    def run(self) -> str:
        # Header + EchoKey opaque gate declarations
        out: List[str] = [
            "OPENQASM 2.0;",
            "include \"qelib1.inc\";",
            "// EchoKey opaque single-qubit gates (canonical axes)",
            "opaque ek_rec(theta) q;   // Y-axis: exp(-i·theta·σ_y)",
            "opaque ek_frac(theta) q;  // Z-axis: exp(-i·theta·σ_z)",
            "",
        ]
        # Registers
        for qreg in self.qc.qregs:
            out.append(f"qreg {qreg.name}[{qreg.size}];")
        for creg in self.qc.cregs:
            out.append(f"creg {creg.name}[{creg.size}];")
        out.append("")

        # Walk instructions (use named attrs to avoid deprecation)
        for ci in self.qc.data:
            inst, qargs, cargs = ci.operation, ci.qubits, ci.clbits
            if is_1q_unitary(inst):
                # accumulate on the single target qubit
                qidx = self._qidx(qargs[0])
                Uop = op_unitary(inst)
                acc = self.acc[qidx]
                acc.U = Uop @ acc.U  # left-multiply for temporal order
                acc.has = True
                self.acc[qidx] = acc
            else:
                # boundary: flush all qubits touched by this inst, then emit it
                for qa in qargs:
                    self._flush_qubit(self._qidx(qa))
                self._emit_instruction(inst, qargs, cargs)

        # End: flush any remaining 1q accumulators
        self._flush_all()

        # Stitch header + body
        return "\n".join(out + self.lines) + "\n"

# -------------------------
# Verification (optional)
# -------------------------

def verify_unitary_prefix(qasm_in: str, qasm_out: str) -> bool:
    """Compare the unitary up to the first measurement/reset. Returns True if fidelities match ~1."""
    # Parse the input circuit
    qc_in = QuantumCircuit.from_qasm_str(qasm_in)
    # Build a verification circuit for the output by replacing ek_* with ZYZ equivalents
    # Since the output QASM declares ek_* as opaque, we rebuild the circuit directly from the
    # emitter's lines: read back and expand ek_* to RZ/RY/RZ in a fresh circuit.
    # For simplicity we just scan the text.
    lines = [ln.strip() for ln in qasm_out.splitlines()]
    # Extract register sizes
    qregs: List[Tuple[str,int]] = []
    for ln in lines:
        if ln.startswith("qreg "):
            name = ln.split()[1].split("[")[0]
            size = int(ln.split("[")[1].split("]")[0].rstrip(";"))
            qregs.append((name, size))
    qc_out = QuantumCircuit(sum(s for _, s in qregs))
    # crude mapping name[idx] → flat index (assumes one qreg or contiguous)
    offsets: Dict[str,int] = {}
    off = 0
    for name, size in qregs:
        offsets[name] = off
        off += size

    def qflat(token: str) -> int:
        nm = token.split("[")[0]
        idx = int(token.split("[")[1].split("]")[0])
        return offsets[nm] + idx

    from qiskit.circuit.library import RZGate, RYGate, CXGate

    for ln in lines:
        if ln.startswith("ek_frac("):
            ang = float(ln.split("(")[1].split(")")[0])
            qtok = ln.split(")")[1].strip(" ;")
            qc_out.append(RZGate(2*ang), [qflat(qtok)])
        elif ln.startswith("ek_rec("):
            ang = float(ln.split("(")[1].split(")")[0])
            qtok = ln.split(")")[1].strip(" ;")
            qc_out.append(RYGate(2*ang), [qflat(qtok)])
        elif ln.startswith("cx "):
            qs = ln[3:].rstrip(";").split(",")
            q0 = qflat(qs[0].strip())
            q1 = qflat(qs[1].strip())
            qc_out.append(CXGate(), [q0, q1])
        elif ln.startswith("measure") or ln.startswith("reset"):
            break  # stop before non‑unitaries
    # Compute fidelities
    U_in = Operator(qc_in.remove_final_measurements(inplace=False)).data
    U_out = Operator(qc_out).data
    F = abs(np.trace(U_in.conj().T @ U_out)) / U_in.shape[0]
    return bool(F > 1 - 1e-10)

# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert QASM2 to EchoKey‑QASM (exact, ZYZ→ek).")
    ap.add_argument("--in", dest="inp", required=True, help="Input OpenQASM 2.0 file")
    ap.add_argument("--out", dest="out", required=True, help="Output EchoKey‑QASM file")
    ap.add_argument("--print", dest="do_print", action="store_true", help="Also print to stdout")
    ap.add_argument("--verify", dest="verify", action="store_true", help="Numerically verify unitary prefix")
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        qasm_in = f.read()
    qc = QuantumCircuit.from_qasm_str(qasm_in)

    emitter = QasmEchoKeyEmitter(qc)
    qasm_out = emitter.run()

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(qasm_out)

    if args.do_print:
        print(qasm_out)

    if args.verify:
        ok = verify_unitary_prefix(qasm_in, qasm_out)
        print(f"[verify] unitary‑prefix fidelity ≈ 1?  {'YES' if ok else 'NO'}")


if __name__ == "__main__":
    main()
