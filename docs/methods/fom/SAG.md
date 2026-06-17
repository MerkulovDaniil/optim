---
title: Stochastic average gradient
parent: First order methods
grand_parent: Methods
order: 6
bibtex: |
  @article{schmidt2017minimizing,
  title={Minimizing finite sums with the stochastic average gradient},
  author={Schmidt, Mark and Le Roux, Nicolas and Bach, Francis},
  journal={Mathematical Programming},
  volume={162},
  number={1-2},
  pages={83--112},
  year={2017},
  publisher={Springer}
  }
file: assets/files/SAG.pdf
---








### Motivation

Consider the general finite-sum minimization problem arising in machine learning:

$$
\min_{x \in \mathbb{R}^p} g(x) := \frac{1}{n} \sum_{i=1}^{n} f_i(x).
$$

Each $f_i$ typically corresponds to the loss on the $i$-th data point. Two classical approaches for solving this problem have complementary strengths and weaknesses.

**Gradient Descent (GD)** computes the exact gradient at every step:

$$
x_{k+1} = x_k - \alpha \cdot \frac{1}{n}\sum_{i=1}^{n} \nabla f_i(x_k).
$$

This gives a reliable descent direction and converges at rate $\mathcal{O}(1/k)$ for convex objectives. However, each iteration requires a full pass over the dataset — prohibitively expensive when $n$ is large.

**Stochastic Gradient Descent (SGD)** replaces the full gradient with a single randomly sampled term:

$$
x_{k+1} = x_k - \alpha \cdot \nabla f_{i_k}(x_k).
$$

Each iteration is cheap ($\mathcal{O}(1)$ gradient evaluations), but the update direction is noisy. As a result, SGD does not converge to the exact minimum for a fixed step size — it oscillates in a neighbourhood of $x^*$. Reducing the step size over time stabilises convergence but slows it down to $\mathcal{O}(1/\sqrt{k})$.

**The key idea of SAG** is to get the best of both worlds. Instead of computing the full gradient 
(expensive) or using only one noisy term (imprecise), SAG maintains a memory table 
$\{y^i\}_{i=1}^n$, where each entry $y^i$ stores the most recently computed gradient of $f_i$:

$$
y^i \leftarrow \nabla f_i(x_k) \quad \text{whenever } i \text{ is sampled.}
$$

At each iteration, a single index $i_k$ is drawn uniformly at random, only $y^{i_k}$ is updated, 
and the step is taken along the average of all stored gradients:

$$
x_{k+1} = x_k - \frac{\alpha}{n} \sum_{i=1}^{n} y_k^i.
$$

The averaged direction $\frac{1}{n}\sum_i y_k^i$ is a biased but low-variance approximation of 
$\nabla g(x_k)$. As iterates progress, the stored gradients $y^i$ become increasingly 
accurate, and the bias vanishes — which is precisely what enables linear convergence.














### Summary

SAG achieves a strictly better convergence rate than SGD while keeping the per-iteration cost
of a single gradient evaluation. The table below summarises the trade-offs:

| | Cost per iteration | Convergence (convex) | Convergence (strongly convex) |
|---|---|---|---|
| **GD** | $\mathcal{O}(n)$ | $\mathcal{O}(1/k)$ | linear |
| **SGD** | $\mathcal{O}(1)$ | $\mathcal{O}(1/\sqrt{k})$ | $\mathcal{O}(1/k)$ |
| **SAG** | $\mathcal{O}(1)$ | $\mathcal{O}(1/k)$ | linear |

The price is **memory**: storing all $n$ gradient vectors requires $\mathcal{O}(np)$ additional space.

**Notable facts**
* The first known result proving linear convergence for a stochastic gradient method in the convex case.
* Convergence bounds carry a factor of $n$; this dependence can be reduced via a restart technique.
* Empirical evaluation is conducted on logistic regression with Tikhonov regularization across multiple datasets.

**Mini-batch SAG**

Just as the single-sample SGD update is rarely used in practice — with mini-batch SGD 
being the default choice — SAG follows the same pattern: the single-index update is the 
canonical form presented in the paper, while the mini-batch version is what one would 
typically reach for in practice. At each step a subset $\mathcal{B}_k \subseteq \{1,\ldots,n\}$ 
of indices is sampled and all corresponding stored gradients are refreshed simultaneously:

$$
y^i_k = \begin{cases} \nabla f_i(x_k), & i \in \mathcal{B}_k \\ y^i_{k-1}, & \text{otherwise} \end{cases}
\qquad
x_{k+1} = x_k - \frac{\alpha}{n} \sum_{i=1}^{n} y^i_k.
$$

The update rule for $x$ remains identical — only more entries of the memory table are refreshed per step. A **non-uniform sampling** variant is also presented, where indices are drawn with probabilities proportional to the individual Lipschitz constants $L_i$, which can improve practical convergence when the $f_i$ are heterogeneous.

### Implementation notes

A naive implementation would recompute $\frac{1}{n}\sum_{i=1}^n y^i_k$ from scratch at every 
step, which costs $\mathcal{O}(n)$ — defeating the purpose. The standard trick is to maintain 
a **running average** and update it incrementally.

The algorithm keeps two objects in memory:
* the full gradient table $\{y^i\}_{i=1}^n \in \mathbb{R}^{n \times p}$,
* the current average $\bar{g} = \frac{1}{n}\sum_{i=1}^n y^i \in \mathbb{R}^p$.

When a mini-batch $\mathcal{B}_k$ is sampled, only the affected entries change. The average 
is updated via a rank-one-style correction — no full recomputation needed:

$$
\bar{g} \leftarrow \bar{g} + \frac{1}{n} \sum_{i \in \mathcal{B}_k} \left(\nabla f_i(x_k) - y^i\right),
$$

then $y^i \leftarrow \nabla f_i(x_k)$ for $i \in \mathcal{B}_k$, and finally:

$$
x_{k+1} = x_k - \alpha \, \bar{g}.
$$

This way each iteration costs $\mathcal{O}(|\mathcal{B}_k|)$ gradient evaluations plus 
$\mathcal{O}(p)$ arithmetic — independent of $n$.

The memory footprint remains $\mathcal{O}(np)$ for the gradient table, which is the 
fundamental limitation of the method.

### Bounds
For a constant step size $\alpha = \dfrac{1}{16 L}$, where $L$ stands for the Lipschitz constant of a gradient of each function $f_i(x)$ (in practice, it means that $L = \max\limits_{i=1, \ldots, n} L_i$).

$$
\mathbb{E}\left[g\left(\overline{x}_{k}\right)\right]-g\left(x^{*}\right) \leqslant \frac{32 n}{k} C_{0},
$$

where $C_0=g\left(x_0 \right)-g\left(x^*\right)+\frac{4L}{n} \| x_0 - x^\ast\|^2 +\frac{\sigma^2}{16L}$ in convex case and

$$
\mathbb{E}\left[g\left(x_{k}\right)\right]-g\left(x^*\right) \leqslant\left(1-\min \left\{\frac{\mu}{16 L}, \frac{1}{8 n}\right\}\right)^{k} C_{0}
$$

in $\mu$ - strongly convex case.
