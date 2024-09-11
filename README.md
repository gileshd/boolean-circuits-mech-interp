# Mechanistic Interpretability of Boolean Circuits

This repository contains research code for investigations into the mechanistic interpretability of neural networks trained to implemented Boolean circuits.

## Installation

Alongside example scripts notebooks this repository contains `boolean_circuits` a python library containing research code.

To install the library clone this repo and run

`pip install -e .`

## Repository Structure

```
.
├── README.md
├── plots/                     \\ scripts for plotting specific results
├── notebooks/                 \\ notebook examples
├── scripts/                   \\ script examples
├── src/
│   └── boolean_circuits/      \\ library source code 
│       ├── circuits.py        \\ implementation of boolean circuits
│       ├── jax/               \\ jax specific code, models, training etc
│       │   ├── data/          \\ code to simulate other boolean data
│       │   ├── circuits.py    \\ jax version of circuits.py (more feature-rich)
│       │   ├── models.py
│       │   ├── probes.py      
│       │   ├── sae.py
│       │   ├── train.py       \\ training code
│       │   └── utils/         \\ misc utilities
│       └── torch/             \\ pytorch specific code
└── tests/                     \\ pytest tests
```

## Multi-task Sparse Parity

Multitask spare parity problem (MSP) from [Michaud et al., (2023)](http://arxiv.org/abs/2303.13506).

## Boolean Circuits

Code to define simulate Boolean circuits.
