# 8 Days of EchoKey — Days 1–8

Tiny, reproducible demos of the **EchoKey** operator family, passes, and math.
Days **1–7** introduce seven single-qubit EchoKey generators that compile exactly to native `RZ/RY/RZ` (**layout-aware**, with fast paths where applicable) and verify **SU(2) closure** plus **emergent Pauli** from a 7-direction frame. Day **8** adds a **QASM→EchoKey** converter and a sparse runner.

## Days 1–7 (super quick)

* **Day 1 — Cyclicity**: `ek_cyc(θ) ≡ e^{-iθ (a₁·σ)}` → exact ZYZ rewrite.
* **Day 2 — Recursion**: `ek_rec(θ) ≡ e^{-iθ (a₂·σ)}` → same rewrite; SU(2) closure checks.
* **Day 3 — Fractality**: `ek_frac(θ) ≡ e^{-iθ σ_z}` with **Z-axis fast path**.
* **Day 4–6**: additional axes/diagonals; multi-qubit demos preserve fidelity ≈ 1.
* **Day 7 — Symplectify**: canonicalization, commutator/anti-commutator numeric checks.
* **Math toy**: from 7 local directions (rows of $A$), the right inverse $B=(A^TA)^{-1}A^T$ yields $S_i=\sum_k B_{ik}E_k^\circ=\sigma_i$ (emergent Pauli), with single-/two-qubit equivalence.

> Everything compiles to basis gates; no opaque `UnitaryGate` left in the IR. Fidelity checks target ≈ 1.0 (global-phase invariant).

---

## Day 8 — QASM→EchoKey (work in progress)

Goal: take an OpenQASM 2.0 circuit; on each wire, fuse every 1-qubit block into at most **three EchoKey ops** using the exact ZYZ identity

$R_z(\alpha)R_y(\beta)R_z(\gamma)\;\mapsto\;
\mathrm{ek\_frac}(\alpha/2)\,\mathrm{ek\_rec}(\beta/2)\,\mathrm{ek\_frac}(\gamma/2)$

We **preserve boundaries** (multi-qubit gates, barriers, resets, measurements, conditionals) and **bitstrings 1-to-1** (no approximations).

### Files

* `day_08_step_01_echokey_QASM.py` — **converter** (QASM2 → EchoKey-QASM2)

  * Declares two opaque gates at the top:

    ```
    opaque ek_rec(theta) q;   // Y-axis: exp(-i·theta·σ_y)
    opaque ek_frac(theta) q;  // Z-axis: exp(-i·theta·σ_z)
    ```
- Fuses supported 1q gates (`u,u2,u3,p,rx,ry,rz,id,x,y,z,h,s,sdg,t,tdg,sx,sxdg`) into ZYZ, then maps
  `RZ(λ) → ek_frac(λ/2)` and `RY(β) → ek_rec(β/2)`.
  * Treats any multi-qubit/conditional/non-unitary as a **flush boundary**.

* `day_08_step_02_echokey_run_sparse_or_compare.py` — **runner**

  * Default: load EchoKey-QASM, expand `ek_*` back to `RZ/RY` for simulation, and run on Aer **sparse** backend (`method=matrix_product_state`).
  * `--compare`: also run the **original** QASM on standard Aer and compare counts (exact match + TVD).

### Install (tested on Qiskit 1.4)

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install "qiskit==1.4.*" "qiskit-aer==0.*" numpy
```

### Step 1 — Convert to EchoKey-QASM

```bash
python day_08_step_01_echokey_QASM.py --in path/to/input.qasm --out path/to/output_ek.qasm --verify --print
```

* `--verify` reconstructs the **unitary prefix** (before first measure/reset) and checks fidelity ≈ 1.
* Output is still **OpenQASM 2.0** with `ek_rec/ek_frac` **opaque** declarations at the top.

### Step 2 — Run sparse (or compare vs raw)

Sparse-only (default):

```bash
python day_08_step_02_echokey_run_sparse_or_compare.py \
  --echokey path/to/output_ek.qasm --shots 8192 --seed 2025
```

Compare vs raw (toggle on):

```bash
python day_08_step_02_echokey_run_sparse_or_compare.py \
  --echokey path/to/output_ek.qasm --compare --normal path/to/input.qasm \
  --shots 8192 --seed 2025
```

Output shows per-backend **transpile**/**execute** timing, top counts, **exact-match** and **TVD**.

### Notes & limits (WIP)

* **QASM2 only.** Some QASM files (especially with non-standard includes/macros) may not parse.
  The runner uses the new `qasm2.loads(...)` with a fallback to `QuantumCircuit.from_qasm_str(...)`.
* **Opaque EchoKey gates.** Simulators don’t know `ek_*`, so the runner expands them to native `RZ/RY` internally. Hardware back-ends or compilers that understand EchoKey can keep them as-is.
* **Conditionals.** `if (c==v) op` are treated as boundaries and preserved verbatim.
* **Bitstrings 1:1.** Conversions are **exact** up to a global phase; final measurement distributions match the original circuit.
* **Performance.** Sparse (`matrix_product_state`) is recommended. Classical-reversible benchmarks (e.g., adders/multipliers with basis-state inputs) are deterministic; consider running **1 shot** or using a boolean evaluator if desired.

### Quick dataset ideas

* **QASMBench** (PNNL) has `small/` (teleportation, grover), `medium/` (qft\_n16, adder\_n25), `large/` (multiplier\_n45 under 60 qubits). Grab a few and run them through the two-step flow.

---

**License:** CC0-1.0 (Public Domain).
If you use this, please cite:
**“8 Days of EchoKey — Day 8: QASM→EchoKey (bitstring-preserving ZYZ fusion)”**

Happy compiling ✨
