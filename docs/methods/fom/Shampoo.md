---
layout: default
title: "Shampoo: Preconditioned Stochastic Tensor Optimization"
parent: First order methods
grand_parent: Methods
nav_order: 9
bibtex: |
  @article{gupta2018shampoo,
  title={Shampoo: Preconditioned stochastic tensor optimization},
  author={Gupta, Vineet and Koren, Tomer and Singer, Yoram},
  journal={arXiv preprint arXiv:1802.09568},
  year={2018}
  }
file: https://arxiv.org/pdf/1802.09568.pdf
---
# Summary
The idea of maintaining second order statistics from accumulated stochastic gradients is the cornerstone of the stochastic first order optimization. Conceptually, guys threat parameter of each layer as a matrix and compute left and right preconditioner instead of one matrix preconditioner to the vectorized parameters, which allows to reduce the number of computations and the amount of memory, required to store.