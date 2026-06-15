---
title: "Gradient Descent vs Proximal Point"
---

The proximal point method is the **implicit Euler** discretization of the gradient flow:
$$
\frac{x_{k+1}-x_k}{\alpha}=-\nabla f(x_{k+1})
\;\Longleftrightarrow\;
x_{k+1}=\operatorname{prox}_{\alpha f}(x_k)=(I+\alpha A)^{-1}x_k .
$$
At the **same** step $\alpha=\alpha^\star_{GD}=2/(\mu+L)$ where gradient descent
already zig-zags, every eigen-factor $1/(1+\alpha\lambda_i)$ of PPM stays in
$(0,1)$, so the proximal iterates approach the minimizer smoothly and monotonically.

:::{.video}
gd_vs_ppm.mp4
:::

[Code](gd_vs_ppm.py)
