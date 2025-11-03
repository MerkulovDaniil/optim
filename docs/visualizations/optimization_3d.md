---
title: "Simple 3D visualization of gradient descent"
---

# Quadratic function

## Gradient descent

Let's start with the simplest case of the gradient descent method applied to a quadratic function:
$$
f(x) = \frac{1}{2} x^T A x - b^T x \qquad x_k+1 = x_k - \alpha_k \nabla f(x_k).
$$

The gradient of the quadratic function is:
$$
\nabla f(x) = A x - b
$$

So the update rule is:
$$
x_{k+1} = x_k - \alpha_k (A x_k - b)
$$

Let's visualize the gradient descent method applied to a quadratic function:

### Strongly convex quadratics

Here we have a strongly convex quadratic function with $\mu = 0.1$ and $L = 3$. Theory claims that we have a convergence both in the objective value and in the domain here.

We can see, that the optimal learning rate $\alpha^* = \frac{2}{\mu + L}$ outperforms other options. While the upper limit for the convergent learning rate is $\frac{1}{L}$.

:::{.video}
opt_3d_SC.mp4
:::

### Isotropic strongly convex quadratics

In this case we have a strongly convex quadratic function with $\mu = L =3$.

It is interesting, that the optimal learning rate gives us convergence exactly to the optimal point in a single step.

:::{.video}
opt_3d_ISC.mp4
:::

### Almost convex quadratics

We lower the $\mu$ to $0.01$, having almost convex (but formally still strongly convex) function.

:::{.video}
opt_3d_ASC.mp4
:::

### Convex quadratics

We lower the $\mu$ to $0$, having convex function. It is very important, that the optimal learning rate is $\frac{1}{L}$ in this case and putting the optimal learning rate for strongly convex case won't give us convergence.

:::{.video}
opt_3d_C.mp4
:::

[Code](optimization_3d.py)