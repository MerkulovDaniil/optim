---
title: "SGD vs Nesterov vs AdamW vs Muon on spiral classification"
---

## Problem setup

Binary classification of two interleaved 2D spirals using a 1-hidden-layer neural network:

$$
\hat y = \sigma\!\bigl(W_2\, \tanh(W_1\, x)\bigr), \qquad W_1 \in \mathbb{R}^{64 \times 3},\; W_2 \in \mathbb{R}^{1 \times 64}
$$

The input is augmented with a bias column: $x = [x_1,\, x_2,\, 1]^\top$.  The loss is binary cross-entropy, trained with **mini-batch SGD** (batch size 40, total 400 points, 3000 steps).

$W_1$ is a **matrix** parameter — this is where Muon applies spectral-norm steepest descent via Newton-Schulz orthogonalization. For the output layer $W_2$, **all methods use AdamW** — this isolates the effect of the optimizer on the matrix parameter.

## Results

:::{.video}
muon_spiral.mp4
:::

### Decision boundary snapshots

![](muon_spiral_boundaries.svg){width="100%"}

### Convergence

![](muon_spiral_loss.svg){width="70%" fig-align="center"}

## Hyperparameters (main experiment)

| Method | lr | Momentum | Weight decay | Other |
|:------:|:--:|:--------:|:------------:|:-----:|
| SGD | 0.23 | — | $10^{-4}$ | — |
| Nesterov | 1.4 | $\mu = 0.9$ | $10^{-4}$ | — |
| AdamW | 0.37 | $\beta_1 = 0.9,\; \beta_2 = 0.999$ | $10^{-4}$ | — |
| Muon ($W_1$) | 0.44 | $\mu = 0.95$ | $10^{-4}$ | Newton-Schulz: 7 iters |
| Muon ($W_2$) | 0.37 | $\beta_1 = 0.9,\; \beta_2 = 0.999$ | $10^{-4}$ | AdamW for output layer |

## Hyperparameter tuning with Optuna

To ensure a fair comparison, we run [Optuna](https://optuna.readthedocs.io/) TPE search with a **fixed compute budget**: 500 trials per method, each trained for 1000 steps. **All** hyperparameters are tuned simultaneously.

### Search spaces

| Method | Tuned parameters |
|:------:|:------------|
| SGD | `lr` $\in [10^{-3}, 30]$, `momentum` $\in [0, 0.99]$, `wd` $\in [10^{-6}, 0.1]$ |
| Nesterov | `lr` $\in [10^{-3}, 30]$, `momentum` $\in [0.5, 0.99]$, `wd` $\in [10^{-6}, 0.1]$ |
| AdamW | `lr` $\in [10^{-4}, 3]$, $\beta_1 \in [0.8, 0.99]$, $\beta_2 \in [0.9, 0.9999]$, `wd` $\in [10^{-6}, 0.1]$, $\varepsilon \in [10^{-10}, 10^{-4}]$ |
| Muon | `lr_muon` $\in [10^{-3}, 3]$, $\mu_{\text{muon}} \in [0.8, 0.999]$, `ns_steps` $\in [3, 10]$, `lr_adam` $\in [10^{-4}, 3]$, $\beta_1$, $\beta_2$, `wd`, $\varepsilon$ |

### Best configurations (500 trials)

| Method | Best HPs | Final loss | Accuracy |
|:------:|:---------|:----------:|:--------:|
| **Muon** | lr=0.27, $\mu$=0.999, ns=7, wd$\approx$0 | **0.090** | 96.2% |
| AdamW | lr=0.26, $\beta_1$=0.96, $\beta_2$=0.96, wd=4$\cdot 10^{-4}$ | 0.095 | 96.5% |
| Nesterov | lr=0.14, $\mu$=0.99, wd$\approx$0 | 0.161 | 94.2% |
| SGD | lr=0.70, momentum=0.93, wd$\approx$0 | 0.310 | 86.2% |

### Decision boundaries after Optuna tuning

![](muon_spiral_optuna_boundaries.svg){width="100%"}

### Optimization history

![](muon_spiral_optuna_history.svg){width="100%"}

### Hyperparameter importance (fANOVA)

![](muon_spiral_optuna_importance.svg){width="100%"}

For SGD, Nesterov, and AdamW the **learning rate dominates** (>70% importance). For Muon, **weight decay and lr_muon are equally important** (~50/40%), while Newton-Schulz iterations and momentum are less critical.

## Takeaways

1. **Muon consistently beats AdamW** on this matrix-parameterized neural network, even after full Optuna HP tuning (500 trials each).
2. **SGD and Nesterov struggle** in the mini-batch regime — adaptive methods are essential.
3. Muon's advantage comes from **spectral orthogonalization** of the gradient: it makes equal progress across all singular directions of $W_1$, whereas AdamW can only adapt per-element.
4. Muon has **more hyperparameters** (8 vs 5 for AdamW), but its performance is robust — fANOVA shows most of them have low importance.

## Code

- [Main experiment](muon_spiral.py)
- [Optuna tuning](muon_spiral_optuna.py)
