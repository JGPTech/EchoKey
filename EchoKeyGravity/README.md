# EchoKey-EFECGSC

**EchoKey-Enhanced Field Equations for Graviton Clustering in Solar-Scale Gravity Cavities**

> **Status:** research prototype (unaudited).
> **License:** **CC0-1.0 (Public Domain)** — “**cite by donating**” (see below).

A single-file simulation that explores a **round-trip quantum↔classical** transition inside an idealized gravity “cavity.” It couples:

* a **TOV-based solar metric** backbone loaded from `solar_metric_data.npz` (`r`, `g00`, `g11`),
* a **graviton-like Schrödinger** evolution,
* lightweight **EchoKey operators** (cyclicity, fractality/recursion, synergy, adaptivity, outlier clamping),
* and **forward + reverse** phases to test classicalization and return.

**Not** a validated GR/QG solver; results are illustrative only.

---

## EchoKey Asks — Compliance (Q/A)

1. **What is this repo, narrowly?**
   A Python script that (1) loads metric components from `solar_metric_data.npz`, (2) evolves quantum & “classical” toy states with EchoKey hooks, (3) logs metrics (overlap, fidelity, entropy, etc.), (4) saves a CSV and a figure.

2. **What problem does it solve here (not in general)?**
   Provides a reproducible sandbox to test whether simple EchoKey-style couplings can steer a state toward a target metric and back, while tracking convergence metrics.

3. **What goes in / what comes out?**
   **Inputs**

* `solar_metric_data.npz` containing arrays: `r`, `g00`, `g11`.
* Optional `initial_conditions.json` (for exact reproducibility).
* Config parameters (see below).

**Outputs**

* `bidirectional_evolution.csv` (per-step metrics).
* `convergence_analysis.png` (9-panel summary plots).
* Optional `initial_conditions.json` (if generated).

4. **One-minute demo (works with your exact code):**

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib pandas

# Verify your NPZ exists and has the expected arrays:
python - << 'PY'
import numpy as np; d=np.load('solar_metric_data.npz')
print(list(d.keys()))        # should show ['r','g00','g11']
print([d[k].shape for k in ['r','g00','g11']])
PY

# Run the simulation (assuming the file below is saved as efecgsc.py)
python efecgsc.py
```

After it finishes you should see:

* `bidirectional_evolution.csv`
* `convergence_analysis.png`
* printed resonance points (overlap peaks > 0.9)

5. **Threat/validation status (plain language)?**

* Simplifying assumptions; no empirical fit/validation claimed.
* No GR/QFT rigor or guarantees; purely exploratory.

6. **Determinism & seeds**
   Set `Config.random_seed` and keep `initial_conditions.json` to reproduce bit-for-bit evolutions (for a fixed NPZ).

7. **EchoKey operator mapping (what’s actually used here)**

| Operator                   | Where in code                                    | Role                                           |
| -------------------------- | ------------------------------------------------ | ---------------------------------------------- |
| **Cyclicity**              | `C_n`, `fractal_potential`                       | periodic terms modulate potentials             |
| **Fractality / Recursion** | `F_n`, `fractal_potential(..., recursion_depth)` | multi-scale shaping of the potential           |
| **Synergy**                | `synergy_matrix`, `compute_synergy_factor`       | cross-state coupling & adaptive scaling        |
| **Adaptivity**             | `adaptive_coupling`, layer-dependent dispersion  | strengthens with overlap; layer-aware dynamics |
| **Outlier clamps**         | explicit clips/bounds in helpers                 | keep evolution bounded/stable                  |

---

## Files & entry point

* **This script** (your pasted code) — please confirm filename (e.g., `efecgsc.py`).
* **`solar_metric_data.npz`** — required; must contain arrays `r`, `g00`, `g11`.
* *(optional)* **`initial_conditions.json`** — created/loaded for reproducibility.

> If you want me to add a small `requirements.txt` and a tiny `make test-run`, say the word and I’ll include them.

---

## Configuration (from `Config`)

```python
layers=3
total_time=200.0     # forward (100) + backward (100)
dt=0.02
dim_mode=64          # spatial resolution
x_min=-5.0; x_max=5.0
mass=1.0
random_seed=98331050
metric_file='solar_metric_data.npz'
initial_conditions_file='initial_conditions.json'
output_csv='bidirectional_evolution.csv'
output_plot='convergence_analysis.png'
load_initial_conditions=False
```

Tune knobs:

* **Resolution/step**: `dim_mode`, `dt`
* **Round-trip length**: `total_time`, `layers`
* **Repro**: `random_seed`, `load_initial_conditions=True`

---

## Input data format (NPZ)

`solar_metric_data.npz` must contain:

* `r`: 1-D radii (meters), strictly increasing
* `g00`: metric $g_{00}(r)$ sampled on `r`
* `g11`: metric $g_{11}(r)$ sampled on `r`

The code interpolates to the simulation grid via cubic splines and **extrapolates** outside `r` (you can tighten that if desired).

> If you want, I can add a **validator** cell/script that checks monotonicity, finite values, and reasonable ranges before running.

---

## Outputs

* **CSV:** `bidirectional_evolution.csv` with columns like:

  * `time`, `q_layer`, `c_layer`, `coupling_strength`, `phase_coherence`,
  * quantum metrics `q_*` and classical metrics `c_*` (`uncertainty_*`, `coherence`, `entropy`, `ipr`, `num_peaks`, `x_expectation`),
  * similarity metrics `similarity_overlap`, `similarity_trace_distance`, `similarity_fidelity`, `similarity_js_divergence`.

* **Figure:** `convergence_analysis.png` (9 panels: uncertainties, coherences, entropies, IPRs, overlap, trace distance, fidelity, JS divergence, layer counts).

* **STDOUT:** “resonance points” where overlap peaks exceed 0.9.

---

## Troubleshooting

* **`FileNotFoundError: solar_metric_data.npz`**
  Put the NPZ in the same directory or set `Config.metric_file` to its path.

* **`KeyError: 'g00'` / wrong arrays**
  Ensure NPZ has **exact** keys: `r`, `g00`, `g11`.

* **NaNs / blow-ups**
  Lower `dt`, reduce `layers`, or reduce recursion depth in `fractal_potential`.

* **No resonance peaks**
  Try longer `total_time`, different `random_seed`, or tweak layer counts.

---

## Non-goals

* Not a substitute for numerical relativity, TOV solvers, or QFT in curved spacetime.
* No empirical claim about real solar graviton phenomena.
* No performance guarantees; it’s a clarity-first reference implementation.

---

## License & “cite by donating”

* **CC0-1.0 — Public Domain.** Use, remix, and redistribute without restriction.
* If this helped you, **please “cite by donating”** to a scientific charity, local mutual-aid, or an open-source maintainer you rely on.

