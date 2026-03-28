---
title: "L1 sparsity: subgradient vs proximal"
---

Consider the LASSO problem:
$$
\min_{x \in \mathbb{R}^n} \frac{1}{2m} \|Ax - b\|_2^2 + \lambda \|x\|_1
$$

The $\ell_1$ regularization promotes sparsity in the solution. However, the method used to solve the problem affects whether the solution components become **exactly zero** or just very small.

**Subgradient method** updates:
$$
x_{k+1} = x_k - \alpha_k g_k, \quad g_k \in \partial \left( f(x_k) + \lambda \|x_k\|_1 \right)
$$
The subgradient of $|x_i|$ at zero is $[-1, 1]$, so subgradient steps almost never land exactly at zero.

**Proximal gradient method** updates:
$$
x_{k+1} = \text{prox}_{\lambda \alpha \|\cdot\|_1}(x_k - \alpha \nabla f(x_k)) = S_{\lambda \alpha}(x_k - \alpha \nabla f(x_k))
$$
where the soft-thresholding operator $S_\kappa(x)_i = \text{sign}(x_i) \cdot [|x_i| - \kappa]_+$ **snaps components to exact zero** when $|x_i| \leq \kappa$.

The animation shows both methods starting from the same random $x_0$ and converging to the same optimal solution $x^*$. The proximal method quickly produces exact zeros (13 out of 20 components), while the subgradient method keeps all components nonzero throughout.

:::{.video}
l1_sparsity.mp4
:::


[Code](l1_sparsity.py)
