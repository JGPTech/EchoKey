# EchoKey-Quantum-Sequencer

A **notebook-based**, quantum-classical toy sequencer that predicts/extends integer sequences using base-10 quantum encoding ideas + classical ML, with light **EchoKey** operator hooks (cyclicity, fractality, synergy, entropy/refraction). **Research only.**

## EchoKey Asks — Compliance (Q/A)

1. **What is this repo, narrowly?**
   A single Jupyter notebook that loads an integer sequence (CSV), applies base-10 encoding + EchoKey-style dynamics, and trains simple ML models to predict the next digits. Plots show predictions; optional CSVs are written from within the notebook.

2. **What problem does it solve here (not in general)?**
   Demonstrates how EchoKey operators can bias/shape a small sequence-prediction pipeline and how base-10 quantum encodings can be prototyped in a classical workflow.

3. **What goes in / what comes out?**

* **Input:** one of the included CSVs (`recaman_puzzle.csv`, `fibonacci_puzzle.csv`, `a003001_puzzle.csv`) with integer sequences.
* **Output:** inline plots in the notebook and (optionally) CSVs the notebook saves (filenames are shown in output cells).

4. **One-minute demo (no boilerplate):**

```bash
# 1) create and activate a venv (optional)
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
# 2) install deps
pip install -r requirements.txt  # or see "Dependencies" below
# 3) start Jupyter
jupyter lab  # or: jupyter notebook
# 4) open EchoKey-Quantum-Sequencer.ipynb and run all cells
#    when prompted in the notebook, pick one CSV (e.g., recaman_puzzle.csv)
```

5. **Threat model & status (plain language)?**
   This repo is **not** a cryptosystem, validator, or production forecaster. No formal guarantees, audits, or statistical significance claims. It is a **prototype** for exploration.

6. **Determinism & seeds—can I reproduce?**
   The notebook exposes seeds in its config cell(s). Set them and re-run all cells with the same CSV to reproduce plots/outputs.

7. **EchoKey operator mapping (what’s actually used here)?**

| Operator             | Where it appears                           | Role in this repo              |
| -------------------- | ------------------------------------------ | ------------------------------ |
| Cyclicity            | periodic terms in feature/gen steps        | seasonal/phase bias in digits  |
| Fractality           | recursive transforms / multi-scale windows | capture self-similar structure |
| Synergy              | rolling-window stats between features      | cross-feature coupling         |
| Entropy / Refraction | noise/temperature & probability “bending”  | regularization / exploration   |

> Note: These are lightweight hooks for experimentation—not a full EchoKey engine.

8. **Actual surfaces you can touch (files):**

* `EchoKey-Quantum-Sequencer.ipynb` — run this end-to-end.
* `Base10Encoding.md` — notes on base-10 encoding ideas used in the notebook.
* `recaman_puzzle.csv`, `fibonacci_puzzle.csv`, `a003001_puzzle.csv` — sample sequences.

9. **How do I change data & knobs (what’s supported)?**
   Open the notebook’s **Config** cell(s) and set: chosen CSV path, train/test split, model choices (e.g., RF, LSTM), window sizes, EchoKey hook strengths (entropy/refraction/synergy), and random seeds.

10. **Non-goals (avoid assumptions):**

* Not a benchmark against SotA sequence models.
* No quantum hardware execution; quantum bits are conceptual/encoded.
* No persistence of a CLI or package API outside the notebook.

11. **License & attribution:**
    MIT (as stated below). Attribution appreciated: **Jon Poplett (JGPTech)**.

---

## Project Structure

```
EchoKey-Quantum-Sequencer/
├── EchoKey-Quantum-Sequencer.ipynb   # main notebook (run this)
├── Base10Encoding.md                 # base-10 encoding notes
├── recaman_puzzle.csv                # sample sequence
├── fibonacci_puzzle.csv              # sample sequence
├── a003001_puzzle.csv                # sample sequence
└── README.md                         # this file
```

## Dependencies

Minimum (use `requirements.txt` if present):

* numpy, pandas, matplotlib
* scikit-learn
* tensorflow (for LSTM experiments)
* pennylane (for encoding experiments)
* jupyterlab or notebook

Install:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow pennylane jupyterlab
```

## Running & Outputs

1. **Launch Jupyter** and open `EchoKey-Quantum-Sequencer.ipynb`.
2. **Run all cells**. Select one of the included CSVs.
3. **See results inline** (predictions/plots). If the notebook writes CSV outputs, it prints the exact path(s) in the output cells.

## Troubleshooting (repo-scoped)

* **Missing packages:** install the deps above.
* **TensorFlow errors on CPU-only:** reduce model size or switch to the RF path in the config cell.
* **Reproducibility:** set seeds in the config cell before running.
* **CSV format:** ensure a single integer column or follow the notebook’s loading cell instructions.

## Contributing

Small, self-contained PRs that improve the **notebook** (repro, clarity, speed, comments) are welcome. Keep scope aligned with this repo.

## License

CC0 - cite by donating. Keep the EchoKey dream alive.
