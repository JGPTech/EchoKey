# 8 Days of EchoKey — Day 1

A tiny, reproducible repo that introduces the **EchoKey** operator family and a
**layout-aware Z–Y–Z rewrite** suitable for the Unitary Compiler Collection (UCC).
It also includes a math walkthrough showing how **Pauli operators emerge** from a
7-operator EchoKey frame.

## Contents

- `echokey_cyclicity.py` — Qiskit 1.4 pass + examples  
  - Defines symbolic `EchoKeyCyclicityGate(θ)`  
  - **Rewrite pass**: `ek_cyc(θ) → RZ(α) · RY(β) · RZ(γ)` using exact SU(2) synthesis  
  - **Layout-aware**: per-wire axis taken from the *physical* qubit after placement  
  - Includes a materializer + fidelity checks (should be ~1.0)

- `ghettomath.py` — standalone NumPy/Scipy toy  
  - Builds a small hex patch (indexing/orientation only)  
  - Instantiates 7 local EchoKey operators per site  
  - Recovers `{σx, σy, σz}` via a right inverse \(B = (A^T A)^{-1} A^T\)  
  - Verifies orthonormality & commutators; single-/two-qubit equivalence

- `GhettoMath.pdf` — LaTeX walkthrough of the **7-operator frame** and emergence

- `8daysofechokey.pdf` — Day-1 pass walkthrough: axis–angle → ZYZ, layout handling, verification

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install qiskit==1.4.0 numpy scipy networkx
````

> Qiskit 1.4 is required for `from qiskit.synthesis import OneQubitEulerDecomposer`.

## Quick Start

### 1) Run the Day-1 pass + demos

```bash
python echokey_cyclicity.py
```

You should see fidelities `≈ 1.000000000000` for all examples, including random batteries.
If you print circuits, you’ll see `ek_cyc` rewritten to `RZ/RY/RZ`.

### 2) Run the math toy (emergent Pauli)

```bash
python ghettomath.py --export_json
```

Expected output:

* All sites: rank(A)=3, orthonormality ≈ identity, commutators ≈ su(2)
* Single-qubit circuit equivalence: fidelity ≈ 1.0
* Two-qubit XYZ equivalence: fidelity ≈ 1.0
* Exports `echokey_pauli_injection.json` (3×7 B-weights per site)

## What is “EchoKey Cyclicity”?

Day-1 defines one symbolic local operator,

$$
\texttt{ek\_cyc}(\theta) \equiv e^{-i\,\theta\,(\mathbf{a}_1\cdot \boldsymbol{\sigma})},
$$

a Bloch rotation by angle $2\theta$ about per-wire axis $\hat{\mathbf a}_1$.
The rewrite pass **exactly** synthesizes the SU(2) unitary into **ZYZ Euler** gates so the
rest of the compiler can optimize, commute, and hardware-target natively.

## Why the 7-operator frame?

At each site we choose 7 unit vectors (rows of \(A\in\mathbb{R}^{7\times 3}\)).
Any EchoKey Hamiltonian
\(H_{\mathrm{EK}}(\mathbf{c})=\sum_k c_k\,(\mathbf{a}_k\cdot\boldsymbol{\sigma})\)
equals \((A^{\top}\mathbf{c})\cdot\boldsymbol{\sigma}\).
When `rank(A)=3`, the **right inverse**
\(B=(A^{\top}A)^{-1}A^{\top}\) yields

\[
S_i \;=\; \sum_k B_{ik}\, E_k^{\circ} \;=\; \sigma_i \, .
\]

so **Pauli emerges** from EchoKey. `ghettomath.py` prints the checks (orthonormality,
commutators, condition numbers).

## Integrating with UCC

* Register the pass after layout (or feed it the final layout) so per-wire axes map to
  **physical** qubits.
* The pass emits basis gates (`RZ`, `RY`, `RZ`) — no opaque `UnitaryGate` in the compiled IR.
* Keep the provided **materializer** in tests to assert unitary equivalence.

## Roadmap

* **Days 2–7**: add 6 more EchoKey operators (new symbolic gates + rewrites).
* **Day 8**: ship the emergence proof (right-inverse construction) and multi-site checks.

## Troubleshooting

* **Fidelity < 1?** Ensure you’re comparing the materialized reference to the pass-only rewrite
  (no extra layout/routing). If running a full pipeline, the pass will follow `final_layout`—
  verify your per-wire axes are keyed by *physical* indices.

## License & Citation

CC0. If you use this, please cite “8 Days of EchoKey — Day 1: Cyclicity → ZYZ”.

---

Happy compiling ✨

```

