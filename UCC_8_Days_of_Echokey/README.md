# 8 Days of EchoKey — Days 1–5

A tiny, reproducible repo that introduces the **EchoKey** operator family and a **layout-aware Z–Y–Z rewrite** suitable for the Unitary Compiler Collection (UCC).
Day 1: **Cyclicity**; Day 2: **Recursion**; Day 3: **Fractality** (with a **Z-axis fast path**); Day 4: **Diagonality-XY**; Day 5: **Diagonality-YZ**. All compile symbolic EchoKey gates to native `RZ/RY/RZ` (or a single `RZ` on Z-aligned axes), verified by unitary-fidelity checks and algebraic identities. A toy script shows how **Pauli operators emerge** from a 7-operator EchoKey frame.

## Contents

* `echokey_cyclicity.py` — Day 1 pass + demos

  * `EchoKeyCyclicityGate(θ)` (k=0, X)
  * **Rewrite:** `ek_cyc(θ) → RZ(α) · RY(β) · RZ(γ)` via exact SU(2)
  * **Layout-aware:** per-wire axis from *physical* qubit
  * Materializer + fidelity checks (≈ 1.0)

* `echokey_recursion.py` — Day 2 pass + demos

  * `EchoKeyRecursionGate(θ)` (k=1, Y)
  * Exact **ZYZ** rewrite; layout-aware; fidelity checks (≈ 1.0)

* `echokey_fractality.py` — Day 3 pass + demos

  * `EchoKeyFractalityGate(θ)` (k=2, Z)
  * **Z-axis fast path:** if axis ≈ ±Z, emit `RZ(±2θ)`; else exact **ZYZ**; fidelity checks (≈ 1.0)

* `echokey_diagonality_xy.py` — Day 4 pass + demos

  * `EchoKeyDiagonalityXYGate(θ)` (k=3; XY diagonal)
  * Exact **ZYZ** rewrite; layout-aware; fidelity checks (≈ 1.0)

* `echokey_diagonality_yz.py` — Day 5 pass + demos

  * `EchoKeyDiagonalityYZGate(θ)` (k=4; YZ diagonal)
  * Exact **ZYZ** rewrite; layout-aware; fidelity checks (≈ 1.0)

* `combined_one_to_five` — **Unified Days 1–5** (stand-alone, walkthrough in comments)

  * `EchoKeyAxisGate(k, θ)` for k∈{0..6}; here we use k=0,1,2,3,4
  * Exact ZYZ rewrite, **Z-fast path**, per-site frames \$A^{(p)}\$
  * Demos: Day 1–5, two-gate `ek_cyc → ek_rec` axis–angle readout, **XYX SU(2) span**, **Lie/Jordan** checks, multi-qubit per-site frames

* `ghettomath.py` — standalone NumPy/SciPy toy

  * Builds a small hex patch; 7 local generators/site
  * Recovers \${\sigma\_x,\sigma\_y,\sigma\_z}\$ via \$B=(A^\top A)^{-1}A^\top\$
  * Orthonormality/commutators; single-/two-qubit equivalence

* PDFs (LaTeX walkthroughs): Day 1 — Cyclicity, Day 2 — Recursion, Day 3 — Fractality, Day 4 — Diagonality (XY), Day 5 — Diagonality (YZ)

## Install

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install qiskit==1.4.0 numpy scipy networkx
```

> Qiskit **1.4** is required for `from qiskit.synthesis import OneQubitEulerDecomposer`.

## Quick Start

### One file (Days 1–5, closure checks + Z-fast)

```bash
python combined_one_to_five.py
```

You’ll see: two-gate `ek_cyc → ek_rec` with axis–angle, Day 1–5 demos (all ≈ 1.0 fidelity), **XYX** SU(2) span, **Lie** $\[a!\cdot!\sigma,b!\cdot!\sigma]=2i(a\times b)!\cdot!\sigma\$ and **Jordan** \${a!\cdot!\sigma,b!\cdot!\sigma}=2(a!\cdot!b)I\$ checks, and multi-qubit per-site frames.

Or run days individually:

```bash
python echokey_cyclicity.py
python echokey_recursion.py
python echokey_fractality.py
python echokey_diagonality.py
python echokey_diagonality_yz.py
```

Math toy:

```bash
python ghettomath.py --export_json
```

Expected: `rank(A)=3`, orthonormality ≈ identity, commutators ≈ su(2); exports `echokey_pauli_injection.json`.

## Day 1–5 generators (math, GitHub-safe)

* **Day-1 (Cyclicity)**
  \$\mathrm{ek\_cyc}(\theta) := \exp!\big(-i,\theta,(\mathbf a\_1!\cdot!\boldsymbol{\sigma})\big)\$ — rotation by \$2\theta\$ about \$\hat{\mathbf a}\_1\$ (≈ X).

* **Day-2 (Recursion)**
  \$\mathrm{ek\_rec}(\theta) := \exp!\big(-i,\theta,(\mathbf a\_2!\cdot!\boldsymbol{\sigma})\big)\$ — about \$\hat{\mathbf a}\_2\$ (≈ Y).

* **Day-3 (Fractality)**
  \$\mathrm{ek\_frac}(\theta) := \exp!\big(-i,\theta,(\mathbf a\_3!\cdot!\boldsymbol{\sigma})\big)\$ — about \$\hat{\mathbf a}\_3\$ (≈ Z).
  **Fast path:** if \$\hat{\mathbf a}\_3\approx\pm\hat z\$, then \$e^{-i\theta(\pm\sigma\_z)} \sim RZ(\pm 2\theta)\$ (global phase ignored).

* **Day-4 (Diagonality-XY)**
  \$\mathrm{ek\_diagxy}(\theta) := \exp!\big(-i,\theta,(\mathbf a\_4!\cdot!\boldsymbol{\sigma})\big)\$ with \$\mathbf a\_4 \propto (1,1,0)\$.

* **Day-5 (Diagonality-YZ)**
  \$\mathrm{ek\_diagyz}(\theta) := \exp!\big(-i,\theta,(\mathbf a\_5!\cdot!\boldsymbol{\sigma})\big)\$ with \$\mathbf a\_5 \propto (0,1,1)\$.

**Exact ZYZ rewrite (phase-equivalent):**

$$
U(\varphi,\hat{\mathbf n})=\cos\!\frac{\varphi}{2}\,I - i\sin\!\frac{\varphi}{2}(\hat{\mathbf n}\!\cdot\!\boldsymbol{\sigma})
\Rightarrow
U = RZ(\alpha)\,RY(\beta)\,RZ(\gamma),
$$

with \$\varphi=2\theta\$ and \$\hat{\mathbf n}=\mathbf a\_k\$.

## SU(2) closure (what we check)

* **Two-gate pair:** `ek_cyc · ek_rec` → prints the **effective axis** and angle of the product.
* **XYX span:** \$\text{ek}[X](\alpha),\text{ek}[Y](\beta),\text{ek}[X](\gamma)\$ composes to arbitrary ZYZ (SU(2)).
* **Lie/Jordan:** errors \~ 0 across examples.

## Integrating with UCC

* Run the pass **after layout** (or provide `final_layout`) so axes read from **physical** wires.
* Emits basis gates (`RZ`, `RY`, `RZ`) — on Z-aligned axes emits a single `RZ`.
* Use the materializer to assert unitary equivalence (fidelity ≈ 1).

## Troubleshooting

* **Fidelity < 1?** Compare the materialized reference against the **pass-only** rewrite (no routing). Ensure \$A^{(p)}\$ is keyed by physical indices and the pass sees `final_layout`.
* **Euler branches:** angles aren’t unique; results are phase-equivalent.
* **Near-identity:** when \$\varphi\to 0\$, any axis is fine; guards are in place.

## License & Citation

**CC0-1.0** (Public Domain).
Please cite: **“8 Days of EchoKey — Days 1–5: Cyclicity, Recursion, Fractality, Diagonality-XY & Diagonality-YZ → ZYZ (layout-aware + Z-fast).”**
