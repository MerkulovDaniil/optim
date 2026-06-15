---
title: "Gradient Descent vs Gradient Flow"
---

Gradient descent is the **explicit Euler** discretization of the gradient flow ODE:
$$
\frac{x_{k+1}-x_k}{\alpha} = -\nabla f(x_k)
\quad\xrightarrow{\;\alpha\to 0\;}\quad
\dot x(t) = -\nabla f\bigl(x(t)\bigr).
$$
On an ill-conditioned quadratic $f(x)=\tfrac12 x^\top A x$ the optimal step
$\alpha^\star = 2/(\mu+L)$ makes GD **zig-zag** across the steep valley, while the
gradient flow follows the smooth continuous trajectory GD is trying to track.

:::{.video}
gd_vs_gf.mp4
:::

[Code](gd_vs_gf.py)
