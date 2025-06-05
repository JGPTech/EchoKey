"""
Setup script for EchoKey Prime Generator
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="echokey-prime",
    version="1.0.0",
    author="Jon Poplett",
    author_email="JonPoplett@JGPTech.net",
    description="A revolutionary prime generator achieving 205 consecutive primes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JGPTech/EchoKey/tree/main/V2Demos",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "sympy>=1.9",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.6b0",
            "flake8>=3.9.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "echokey-prime=echokey_prime:main",
        ],
    },
)