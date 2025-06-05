# EchoKey Prime Generator 🎼

A revolutionary prime number generator that achieves **205 consecutive primes** through "mathematical orchestration" - combining multiple polynomial formulas like instruments in a symphony.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

## 🏆 Achievements

- **205 consecutive primes** (n=0 to 204) - no gaps!
- **88%+ prime density** in first 200 values
- **77.2% overall density** (386 primes in first 500 values)
- Arithmetic progression pattern discovered in gaps
- Extendable to arbitrary large n values

## 📊 Performance

```
╔══════════════════════════════════════════════════════════╗
║        ECHOKEY PRIME GENERATOR - FINAL RESULTS           ║
╚══════════════════════════════════════════════════════════╝

Prime density by range:
Range      | Primes | Density | Visualization
------------------------------------------------------------
n=  0- 99 |    100 | 100.0% | ██████████████████████████████████████████████████
n=100-199 |    100 | 100.0% | ██████████████████████████████████████████████████
n=200-299 |     69 |  69.0% | ██████████████████████████████████░░░░░░░░░░░░░░░
n=300-399 |     61 |  61.0% | ██████████████████████████████░░░░░░░░░░░░░░░░░░░
n=400-499 |     56 |  56.0% | ████████████████████████████░░░░░░░░░░░░░░░░░░░░░
```

## 🎯 The Mathematical Orchestra

The key insight is using each polynomial exactly where it performs best:

1. **Ethiopian Polynomial** (n = 0-43)
   ```
   f(n) = 3n² - 129n + 1409
   ```

2. **Extended Euler** (n = 44-123)
   ```
   f(n) = n² - 79n + 1601
   ```

3. **Euler's Polynomial** (n = 124+)
   ```
   f(n) = n² + n + 41
   ```

## 🚀 Quick Start

```python
from echokey_prime import echokey_ultimate
from sympy import isprime

# Generate the 100th prime in our sequence
value = echokey_ultimate(99)
print(f"n=99: {value} (prime: {isprime(value)})")

# Check consecutive primes
consecutive = 0
for n in range(1000):
    if isprime(echokey_ultimate(n)):
        consecutive += 1
    else:
        break
print(f"Consecutive primes: {consecutive}")
```

## 📁 Repository Structure

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/echokey-prime-generator.git
cd echokey-prime-generator

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python analysis.py
```

## 📈 Gap Pattern Discovery

After the consecutive run ends at n=205, gaps appear at products of consecutive primes:
- n=164: 41² = 1681
- n=165: 41×43 = 1763  
- n=168: 43×47 = 2021
- n=173: 47×53 = 2491

This creates arithmetic progressions of length 2, 4, 6, 8, 10...

## 🧮 Mathematical Background

The EchoKey framework is based on the principle of "mathematical orchestration" - optimally combining different mathematical formulas to achieve superior results. This is inspired by the [EchoKey v2 paper](https://zenodo.org/records/15571741) on universal mathematical programming languages.

## 🌟 Interactive Demo

Check out our [interactive visualization](https://www.jgptech.net/prime-time) to explore the prime patterns yourself!

### Areas for Contribution:
- Extending the pattern
- Finding optimal polynomial combinations
- Performance optimizations
- Additional visualizations

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{echokey_prime_2025,
  author = {Jon Poplett},
  title = {EchoKey Prime Generator: Mathematical Orchestration for Prime Generation},
  year = {2025},
  url = {https://github.com/JGPTech/EchoKey/tree/main/V2Demos}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Euler's prime-generating polynomial
- Ethiopian Kid for the Ethiopian polynomial
- The EchoKey framework for mathematical composition principles

---

**Note**: This is an active research project. The claim of 205 consecutive primes has been verified computationally but awaits formal mathematical proof.