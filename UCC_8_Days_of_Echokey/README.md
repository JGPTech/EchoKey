# 8 Days of EchoKey — Days 1–6

A tiny, reproducible repo introducing the **EchoKey** operator family and a **layout-aware Z–Y–Z rewrite** suitable for the Unitary Compiler Collection (UCC).
Day 1: **Cyclicity**; Day 2: **Recursion**; Day 3: **Fractality** (with a **Z-axis fast path**); Day 4: **Diagonality-XY**; Day 5: **Diagonality-YZ**; Day 6: **Diagonality-XZ**. All compile symbolic EchoKey gates to native `RZ/RY/RZ` (or a single `RZ` on Z-aligned axes) and are verified by fidelity checks and algebraic identities. A toy script shows how **Pauli operators emerge** from a 7-operator EchoKey frame.

## Contents

* Day files (pass + demos + PDF walkthrough)

  * `day_01_echokey_cyclicity.py` · `day_01_echokey_cyclicity.pdf` (k=0, X)
  * `day_02_echokey_recursion.py` · `day_02_echokey_recursion.pdf` (k=1, Y)
  * `day_03_echokey_fractality.py` · `day_03_echokey_fractality.pdf` (k=2, Z; **Z-fast path**)
  * `day_04_echokey_diagonality.py` · `day_04_echokey_diagonality.pdf` (k=3, XY diagonal)
  * `day_05_echokey_diagonality_Yz.py` · `day_05_echokey_diagonality_Yz.pdf` (k=4, YZ diagonal)
  * `day_06_echokey_diagonality_Xz.py` · `day_06_echokey_diagonality_Xz.pdf` (k=5, XZ diagonal)

* Unified (comment-walkthrough) runners

  * `combined_one_two.py` (Days 1–2)
  * `combined_one_to_three.py` (Days 1–3)
  * `combined_one_to_four.py` (Days 1–4)
  * `combined_one_to_five.py` (Days 1–5)
  * `combined_one_to_six.py` (Days 1–6, current)

* Math toy (emergent Pauli)

  * `ghettomath.py` · `ghettomath.pdf`

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

### One file (Days 1–6, closure checks + Z-fast)

```bash
python combined_one_to_six.py
```

You’ll see: two-gate `ek_cyc → ek_rec` with an axis–angle printout; Day 1–6 demos (≈ 1.0 fidelity each); **XYX** SU(2) span; **Lie** $\[a!\cdot!\sigma,,b!\cdot!\sigma]=2i(a\times b)!\cdot!\sigma\$ and **Jordan** \${a!\cdot!\sigma,,b!\cdot!\sigma}=2(a!\cdot!b)I\$ checks; multi-qubit per-site frames.

Or run days individually:

```bash
python day_01_echokey_cyclicity.py
python day_02_echokey_recursion.py
python day_03_echokey_fractality.py
python day_04_echokey_diagonality.py
python day_05_echokey_diagonality_Yz.py
python day_06_echokey_diagonality_Xz.py
```

Math toy:

```bash
python ghettomath.py --export_json
```

Expected: `rank(A)=3`, orthonormality ≈ identity, commutators ≈ su(2); exports `echokey_pauli_injection.json`.

## What are the Day 1–6 generators?

* **Day-1 (Cyclicity)**
  \$\mathrm{ek\_cyc}(\theta)=\exp!\big(-i,\theta,(\mathbf a\_1!\cdot!\boldsymbol{\sigma})\big)\$ — rotate by \$2\theta\$ about \$\hat{\mathbf a}\_1\$ (≈ X).

* **Day-2 (Recursion)**
  \$\mathrm{ek\_rec}(\theta)=\exp!\big(-i,\theta,(\mathbf a\_2!\cdot!\boldsymbol{\sigma})\big)\$ — about \$\hat{\mathbf a}\_2\$ (≈ Y).

* **Day-3 (Fractality)**
  \$\mathrm{ek\_frac}(\theta)=\exp!\big(-i,\theta,(\mathbf a\_3!\cdot!\boldsymbol{\sigma})\big)\$ — about \$\hat{\mathbf a}\_3\$ (≈ Z).
  **Fast path:** if \$\hat{\mathbf a}\_3\approx\pm\hat z\$, then \$e^{-i\theta(\pm\sigma\_z)} \sim RZ(\pm 2\theta)\$ (ignore global phase).

* **Day-4 (Diagonality-XY)**
  \$\mathrm{ek\_diagxy}(\theta)=\exp!\big(-i,\theta,(\mathbf a\_4!\cdot!\boldsymbol{\sigma})\big)\$ with \$\mathbf a\_4\propto(1,1,0)\$.

* **Day-5 (Diagonality-YZ)**
  \$\mathrm{ek\_diagyz}(\theta)=\exp!\big(-i,\theta,(\mathbf a\_5!\cdot!\boldsymbol{\sigma})\big)\$ with \$\mathbf a\_5\propto(0,1,1)\$.

* **Day-6 (Diagonality-XZ)**
  \$\mathrm{ek\_diagxz}(\theta)=\exp!\big(-i,\theta,(\mathbf a\_6!\cdot!\boldsymbol{\sigma})\big)\$ with \$\mathbf a\_6\propto(1,0,1)\$.

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
Please cite: **“8 Days of EchoKey — Days 1–6: Cyclicity, Recursion, Fractality, Diagonality-XY, Diagonality-YZ & Diagonality-XZ → ZYZ (layout-aware + Z-fast).”**
