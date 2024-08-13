# Mechanistic Interpretability of Boolean Circuits

This repository contains research code for investigations into the mechanistic interpretability of neural networks trained to implemented Boolean circuits.


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
│       │   ├── models.py
│       │   ├── sae.py
│       │   └── utils/         \\ misc utilities
│       └── torch/             \\ pytorch specific code
└── tests/                     \\ pytest tests
```

## Multi-task Sparse Parity

Multitask spare parity problem (MSP) from [Michaud et al., (2023)](http://arxiv.org/abs/2303.13506).

## Boolean Circuits

Code to define simulate Boolean circuits.
