# 8 Days of EchoKey — Days 1–3

A tiny, reproducible repo that introduces the **EchoKey** operator family and a **layout-aware Z–Y–Z rewrite** suitable for the Unitary Compiler Collection (UCC).
Day 1 covers **Cyclicity**; Day 2 adds **Recursion**; Day 3 adds **Fractality** with a **Z-axis fast path**. All compile symbolic EchoKey gates to native `RZ/RY/RZ` with exact SU(2) synthesis (or a single `RZ` on Z-aligned axes), verified by unitary-fidelity checks and algebraic identities. A toy script shows how **Pauli operators emerge** from a 7-operator EchoKey frame.

## Contents

* `echokey_cyclicity.py` — Day 1 pass + demos

  * `EchoKeyCyclicityGate(θ)` (k=0)
  * **Rewrite:** `ek_cyc(θ) → RZ(α) · RY(β) · RZ(γ)` via exact SU(2)
  * **Layout-aware:** per-wire axis pulled from *physical* qubit after placement
  * Materializer + fidelity checks (expect ≈ 1.0)

* `echokey_recursion.py` — Day 2 pass + demos

  * `EchoKeyRecursionGate(θ)` (k=1)
  * Same **ZYZ rewrite** + **layout-aware** axis resolution
  * Materializer + fidelity checks (expect ≈ 1.0)

* `echokey_fractality.py` — Day 3 pass + demos

  * `EchoKeyFractalityGate(θ)` (k=2)
  * **Z-axis fast path:** when the resolved axis is ≈ ±Z, emit `RZ(±2θ)`; otherwise exact **ZYZ**
  * Layout-aware + fidelity checks (expect ≈ 1.0)

* `echokey_axisgate_123.py` — **Unified Days 1–3** (stand-alone, walkthrough in comments)

  * `EchoKeyAxisGate(k, θ)` for k∈{0..6}; Days 1–3 use k=0,1,2
  * Exact ZYZ rewrite, Z-fast path, and per-site frames \$A^{(p)}\$
  * Demos: Day 1–3, **two-gate pair** (`ek_cyc → ek_rec`) axis–angle readout, **XYX SU(2) span**, **Lie**/**Jordan** checks, multi-qubit per-site frames

* `ghettomath.py` — standalone NumPy/SciPy toy

  * Builds a small hex patch (indexing/orientation only)
  * Instantiates 7 local EchoKey generators per site
  * Recovers \${\sigma\_x,\sigma\_y,\sigma\_z}\$ via right inverse \$B=(A^\top A)^{-1}A^\top\$
  * Verifies orthonormality & commutators; single-/two-qubit circuit equivalence

* Day docs (PDF): “Day 1 — Cyclicity”, “Day 2 — Recursion”, “Day 3 — Fractality”

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

### A) One file to see it all (Days 1–3, closure checks + Z fast path)

```bash
python echokey_axisgate_123.py
```

You’ll see:

* **Two-gate pair** `ek_cyc → ek_rec`: fidelity ≈ 1.0 and a printed **effective axis–angle**
* Day 1–3 demos: fidelities ≈ 1.0; `ek_axis` rewritten to `RZ/RY/RZ` (or `RZ` on Z-fast path)
* **XYX** composition (k=0,1,0) showing practical SU(2) span
* **Lie** $\[a!\cdot!\sigma, b!\cdot!\sigma]=2i(a\times b)!\cdot!\sigma\$ → error \~ 0
* **Jordan** \${a!\cdot!\sigma, b!\cdot!\sigma}=2(a!\cdot!b)I\$ → error \~ 0
* Multi-qubit per-site frames + CX chain → fidelity ≈ 1.0

### B) Day 1 only

```bash
python echokey_cyclicity.py
```

### C) Day 2 only

```bash
python echokey_recursion.py
```

### D) Day 3 only

```bash
python echokey_fractality.py
```

### E) Emergent Pauli (math toy)

```bash
python ghettomath.py --export_json
```

Expected:

* All sites: `rank(A)=3`, orthonormality ≈ identity, commutators ≈ su(2)
* Single-qubit and two-qubit circuit equivalences: fidelity ≈ 1.0
* Exports `echokey_pauli_injection.json` (3×7 B-weights per site)

## What are “Cyclicity”, “Recursion”, and “Fractality”?

**Day-1**

$$
\mathrm{ek\_cyc}(\theta) := \exp\!\big(-i\,\theta\,(\mathbf a_1\!\cdot\!\boldsymbol{\sigma})\big),
$$

a Bloch rotation by angle \$2\theta\$ about per-wire axis \$\hat{\mathbf a}\_1\$ (≈ X).

**Day-2**

$$
\mathrm{ek\_rec}(\theta) := \exp\!\big(-i\,\theta\,(\mathbf a_2\!\cdot\!\boldsymbol{\sigma})\big),
$$

a Bloch rotation by angle \$2\theta\$ about per-wire axis \$\hat{\mathbf a}\_2\$ (≈ Y).

**Day-3**

$$
\mathrm{ek\_frac}(\theta) := \exp\!\big(-i\,\theta\,(\mathbf a_3\!\cdot\!\boldsymbol{\sigma})\big),
$$

a Bloch rotation by angle \$2\theta\$ about per-wire axis \$\hat{\mathbf a}\_3\$ (≈ Z).
**Fast path:** if \$\hat{\mathbf a}\_3\approx\pm\hat z\$, then

$$
e^{-i\theta(\pm\sigma_z)} \;\sim\; RZ(\pm 2\theta)\,,
$$

(up to a global phase).

All rewrite **exactly** to ZYZ Euler gates (up to global phase):

$$
U(\varphi,\hat{\mathbf n})=\cos\!\frac{\varphi}{2}\,I - i\sin\!\frac{\varphi}{2}(\hat{\mathbf n}\!\cdot\!\boldsymbol{\sigma})
\;\Rightarrow\;
U = RZ(\alpha)\,RY(\beta)\,RZ(\gamma),
$$

with \$\varphi=2\theta\$ and \$\hat{\mathbf n}=\mathbf a\_k\$ (k=1,2,3 respectively).

## Why the 7-operator frame?

Pick 7 unit vectors (rows of \$A\in\mathbb R^{7\times 3}\$) at each site. Any EchoKey Hamiltonian
\$H\_{\rm EK}(\mathbf c)=\sum\_k c\_k,(\mathbf a\_k!\cdot!\boldsymbol{\sigma})\$
equals \$(A^\top\mathbf c)!\cdot!\boldsymbol{\sigma}\$.
If \$\mathrm{rank}(A)=3\$, the **right inverse** \$B=(A^\top A)^{-1}A^\top\$ yields

$$
S_i=\sum_k B_{ik}\,E_k^\circ=\sigma_i,
$$

so **Pauli emerges** from EchoKey. The toy script prints orthonormality, commutators, and condition numbers.

## SU(2) closure (what we check)

* **Two-gate pair:** `ek_cyc · ek_rec` → prints the **effective axis** and angle of the product.
* **XYX span:** \$\text{ek}[X](\alpha),\text{ek}[Y](\beta),\text{ek}[X](\gamma)\$ composes to arbitrary ZYZ (hence SU(2)).
* **Lie commutator:** $\[a!\cdot!\sigma, b!\cdot!\sigma]=2i(a\times b)!\cdot!\sigma\$ → numeric error \~ 0.
* **Jordan anti-commutator:** \${a!\cdot!\sigma, b!\cdot!\sigma}=2(a!\cdot!b)I\$ → numeric error \~ 0.

## Integrating with UCC

* Register the pass **after layout** (or give it `final_layout`) so axes read from **physical** wires.
* The pass emits basis gates (`RZ`, `RY`, `RZ`) — no opaque `UnitaryGate` in your compiled IR; on Z-aligned axes it emits a single `RZ`.
* Keep the provided **materializer** in tests and assert unitary equivalence (fidelity ≈ 1).

## Troubleshooting

* **Fidelity < 1?** Compare the materialized reference to the **pass-only** rewrite (no extra mapping/routing). If you run a full pipeline, verify your per-wire frames \$A^{(p)}\$ are keyed by *physical* indices and the pass sees `final_layout`.
* **Euler branches:** ZYZ angles aren’t unique; different branches are **globally phase-equivalent**.
* **Near-identity:** when \$\varphi\to 0\$, any axis is fine; code guards are in place.

## Roadmap

* **Days 4–7:** add the remaining EchoKey generators (symbolic gates + ZYZ rewrites / fast paths as applicable).
* **Day 8:** ship the full emergence proof pack (right-inverse construction, multi-site checks).

## License & Citation

**CC0-1.0** (Public Domain).
If you use this, please cite:
**“8 Days of EchoKey — Days 1–3: Cyclicity, Recursion & Fractality → ZYZ (layout-aware + Z-fast).”**

---

Happy compiling ✨
