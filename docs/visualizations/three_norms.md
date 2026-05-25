---
title: "Steepest descent in three norms: SGD vs SignGD vs Muon"
---

## Setup

Ill-conditioned matrix quadratic with condition number $\kappa = 20$:

$$
f(W) = \frac{1}{2}\|\Sigma W\|_F^2, \qquad W \in \mathbb{R}^{8 \times 8}.
$$

Three optimizers correspond to steepest descent in three different norms:

| Optimizer | Norm | Update direction |
|:---------:|:----:|:----------------:|
| **SGD** | Frobenius $\|\cdot\|_F$ | $-G / \|G\|_F$ |
| **SignGD** | Elementwise $\ell_\infty$ | $-\text{sign}(G)$ |
| **Muon** | Spectral $\|\cdot\|_\sigma$ | $-UV^\top$ (polar factor) |

Learning rates are individually tuned via grid search for each method.

## Animation

:::{.video}
three_norms.mp4
:::

**Left panel**: convergence of $f(W_k) / f(W_0)$ in log scale. All three methods converge, but at different rates depending on how well the norm matches the problem geometry.

**Right panel**: normalized singular value spectrum of the update direction $\Delta W$ at each iteration. SGD concentrates almost all energy on the top 1--2 singular directions (the dominant eigenvectors of $\Sigma^2$). Muon distributes energy uniformly across all directions — every singular value of $UV^\top$ equals 1 by construction.

## Key observation

The spectral structure of the update reveals *why* the methods differ:

- **SGD** is biased toward the directions where the gradient is largest (high-curvature directions). Low-curvature directions receive negligible updates.
- **SignGD** normalizes per-element, partially compensating for scale differences, but does not account for correlations between coordinates.
- **Muon** normalizes the entire matrix spectrally — all singular directions get equal weight. This is equivalent to steepest descent in the spectral norm, or equivalently, the LMO over the unit ball of the nuclear norm.

## Code

- [Experiment script](three_norms.py)
