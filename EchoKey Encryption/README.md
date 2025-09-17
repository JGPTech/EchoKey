# EchoKey Encryption System

A research-grade, round-trip **byte stream transformer** that explores EchoKey operators (cyclicity, recursion/fractality, synergy, outliers) as **keyed stream-cipher–style** transformations over data. Includes deterministic keystream (HMAC-SHA256) + flip-map permutation + oscillator/rolling-window dynamics. **Not audited. Do not use for protecting sensitive data.**

## EchoKey Asks — Compliance (Q/A)

1. **What is this repo, narrowly?**
   A Python package + CLI that encrypts/decrypts bytes using a keystream (HMAC-SHA256) and EchoKey-style dynamic transforms (flip map, oscillators, synergy windows). Round-trip is verified with tests.

2. **What problem does it solve here (not in general)?**
   Demonstrates how EchoKey operators can be embedded in a reversible, high-throughput byte pipeline with deterministic seeding and reproducible runs.

3. **What goes in / what comes out?**
   **Input:** bytes (file or text). **Output:** ciphertext (bytes) + a sidecar of per-batch encrypted keys (for decryption). Decryption returns original bytes.

4. **Single command demo (<60s)?**

```bash
# create a venv (optional)
python -m venv .venv && . .venv/Scripts/activate  # Windows
# or: source .venv/bin/activate                    # macOS/Linux

pip install -r requirements.txt
python -m echokey_enc --encrypt --in README.md --out README.md.enc
python -m echokey_enc --decrypt --in README.md.enc --out README.dec.md
# diff should be empty if round-trip succeeded
```

5. **Threat model & security status (plain language)?**

* Uses **HMAC-SHA256** for keystream blocks + entropy mask.
* Adds experimental transforms (flip map, oscillators, synergy windows).
* No formal proofs, no cryptanalysis, no side-channel audit.
  ➡️ **Research only.** Do **not** use for production/PII/financial/health data.

6. **Determinism & seeds—how to reproduce exactly?**

* Global seed is persisted in `random_seed.txt` (5000-digit) or a 256-bit runtime seed.
* Keystream is deterministic given `(secret_key, seed, counter)`.
* Flip map is derived from the seed; inverse is stored for decryption.
* Set `--secret-key` or `EKEY_SECRET` and keep `random_seed.txt` to reproduce bit-identical outputs.

7. **EchoKey operator mapping (what’s used, where)?**
   \| Operator | Where it lives | What it does here |
   \|---|---|---|
   \| Cyclicity | `StateEvolver.compute_cyclic_function` | Periodic modulation over position/time |
   \| Recursion/Fractality | `FractalGenerator.generate` | Nested transform of cyclic term |
   \| Synergy | `SynergyCalculator` / `calculate_synergy_numba` | Cross-window coupling for parameter updates |
   \| Outliers | `StateEvolver.outlier_term` (stub) | Hook to bias dynamics under anomalies |
   \| Interdependence | rolling windows (x/y/acoustic) | Windows feed each other’s updates |
   \| Nonlinearity | tanh/atan/sin clamps in Numba kernel | Keeps dynamics bounded and reversible |
   \| Adaptivity | parameter evolution per byte | Updates α/β/ω within safe bounds |

8. **Exact API surface (what can I import)?**

```python
from echokey_enc.core import EchoKeyEncryption
from echokey_enc.keys import KeystreamScrambler
from echokey_enc.flipmap import generate_flip_map, invert_flip_map, randomized_character_flip
# Utilities live under: echokey_enc/numba_kernels.py, echokey_enc/state.py, echokey_enc/synergy.py
```

9. **CLI surface (flags that actually exist here)?**

```bash
python -m echokey_enc --encrypt --in <path> --out <path> [--secret-key hex|str] [--batch-size N] [--debug]
python -m echokey_enc --decrypt --in <path> --out <path> [--secret-key hex|str] [--batch-size N] [--debug]
python -m echokey_enc --gen-test-data --zeros <path> --random <path> --n 1048576
python -m echokey_enc --self-test          # single-byte and round-trip sanity checks
```

10. **Performance knobs (what should I tune)?**

* `BATCH_SIZE` (default 102\_400) — throughput vs memory.
* `NUMBA` JIT warms on first call; reruns are fast.
* Windows `WINDOW_SIZE` (default 8) — affects synergy dynamics cost minimally.

11. **Non-goals (so reviewers don’t assume them)?**

* No authenticated encryption (AEAD) or MAC over ciphertext.
* No standard cipher compatibility (AES/ChaCha20).
* No key exchange, KDF, or protocol framing.
* No side-channel, constant-time, or hardware-accel guarantees.

12. **License & attribution?**
    Code and docs are **CC0-1.0** (Public Domain). Attribution appreciated: **Jon Poplett (JGPTech)**.

---

## Install

```bash
pip install -r requirements.txt
# requirements.txt should minimally include:
# numpy
# numba
# tqdm
```

---

## Quick Start (Library)

```python
from echokey_enc.core import EchoKeyEncryption

ek = EchoKeyEncryption(
    seed=None,                 # 256-bit random if None; or pass int
    window_size=8,
    batch_size=102_400,
    debug=False,
    secret_key=b"demo-secret"  # set from env for real runs
)

plaintext = b"hello echokey"
cipher = ek.encrypt(plaintext)
roundtrip = ek.decrypt(cipher)
assert roundtrip == plaintext
```

---

## How it Works (repo-scoped)

1. **Flip Map (seeded, reversible)**
   `generate_flip_map(seed)` produces a 16×16 permutation of byte values; applied nibble-wise via `randomized_character_flip`. Inverse map guarantees round-trip.

2. **Keystream (HMAC-SHA256)**
   `KeystreamScrambler(seed, secret_key)` emits 32-byte blocks keyed by `(secret_key, f"{seed}:{counter}")`. Keystream is XORed with data plus an entropy mask `SHA256(keystream)`.

3. **Dynamic Pass (Numba)**
   `process_batch_numba` applies bounded oscillator updates and parameter evolution (α, β, ω) driven by synergy computed from rolling windows. All updates are clamped to preserve reversibility.

4. **Batching & Keys**
   Each batch evolves internal state and appends an encrypted per-batch key record (used only for decryption alignment). Files store ciphertext + sidecar.

---

## Reproducibility & Logging

* **Seeds:** `random_seed.txt` (or pass a 256-bit integer).
* **Debug logs:** `logs/encryption_debug_YYYYMMDD-HHMMSS.log` when `--debug` or `DEBUG_MODE=True`.
* **Determinism:** repeat runs with the **same** seed, secret key, and inputs produce identical ciphertext.

---

## Tests

```bash
# sanity:
python -m echokey_enc --self-test

# round-trip on generated data:
python -m echokey_enc --gen-test-data --zeros zero.bin --random rand.bin --n 1048576
python -m echokey_enc --encrypt --in rand.bin --out rand.bin.enc
python -m echokey_enc --decrypt --in rand.bin.enc --out rand.bin.dec
fc rand.bin rand.bin.dec   # (Windows) or: diff -s rand.bin rand.bin.dec
```

---

## Configuration (defaults that actually exist)

```python
SEED_FILE = "random_seed.txt"
WINDOW_SIZE = 8
BATCH_SIZE = 102_400
DEBUG_MODE = False

PARAMS_ALPHA_INITIAL = 0.009
PARAMS_BETA_INITIAL  = 0.002
PARAMS_OMEGA_INITIAL = 0.006

SYNERGY_DIMENSIONS = 3
KAPPA_MATRIX_SEED  = 42
FRACTAL_LEVELS     = 5
FRACTAL_BASE_CONSTANT = 0.4
MULTIDIMENSIONAL_DIMENSIONS = 3
OUTLIER_THRESHOLD  = 0.1
OUTLIER_WEIGHT     = 1.0
```

> These constants are **repo-scoped** and used exactly as implemented; remove or change them only if you update the corresponding code paths.

---

## Extending (only what’s supported)

* Plug your own `outlier_term(t)` into `StateEvolver`.
* Replace `base_function` in `FractalGenerator` to test alternative recursive transforms.
* Override `functions` in `SynergyCalculator` to change component influences.
* Keep clamps/bounds to preserve reversibility.

---

## Safety & Ethics

This is experimental cryptography. **Do not** depend on it for confidentiality, integrity, or authenticity. If you publish results, state clearly: *“EchoKey Encryption System is research-grade and unaudited.”*

---

## Maintainers

**Jon Poplett (JGPTech)** — CC0-1.0. PRs welcome if they stay within scope (encryption demo + operator mapping).
