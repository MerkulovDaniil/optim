---
layout: default
title: Inexact line search
parent: Line search
grand_parent: Methods
---

This strategy of inexact line search works well in practice, as well as it has the following geometric interpretation:

# Sufficient decrease

Let's consider the following scalar function while being at a specific point of $$x_k$$: 

$$
\phi(\alpha) = f(x_k - \alpha\nabla f(x_k)), \alpha \geq 0
$$

consider first order approximation of  $$\phi(\alpha)$$:

$$
\phi(\alpha) \approx f(x_k) - \alpha\nabla f(x_k)^\top \nabla f(x_k)
$$

A popular inexact line search condition stipulates that $$\alpha$$ should first of all give sufficient decrease in the objective function $$f$$, as measured by the following inequality:

$$
f(x_k - \alpha \nabla f (x_k)) \leq f(x_k) - c_1 \cdot \alpha\nabla f(x_k)^\top \nabla f(x_k)
$$

for some constant $$c_1 \in (0,1)$$. (Note, that $$c_1 = 1$$ stands for the first order Taylor approximation of $$\phi(\alpha)$$). This is also called Armijo condition. The problem of this condition is, that it could accept arbitrary small values $$\alpha$$, which may slow down solution of the problem. In practice, c1 is chosen to be quite small, say $$c_1 \approx 10^{−4}$$.

# Curvature condition

To rule out unacceptably short steps one can introduce a second requirement:

$$
-\nabla f (x_k - \alpha \nabla f(x_k))^\top \nabla f(x_k) \geq c_2 \nabla f(x_k)^\top(- \nabla f(x_k))
$$

for some constant $$c_2 \in (c_1,1)$$, where $$c_1$$ is a constant from Armijo condition. Note that the left-handside is simply the derivative $$\nabla_\alpha \phi(\alpha)$$, so the curvature condition ensures that the slope of $$\phi(\alpha)$$ at the target point is greater than $$c_2$$ times the initial slope $$\nabla_\alpha \phi(\alpha)(0)$$. Typical values of $$c_2 \approx 0.9$$ for Newton or quasi-Newton method. The sufficient decrease and curvature conditions are known collectively as the Wolfe conditions.

# Goldstein conditions

Let's consider also 2 linear scalar functions $$\phi_1(\alpha), \phi_2(\alpha)$$:

$$
\phi_1(\alpha) = f(x_k) - \alpha \alpha \|\nabla f(x_k)\|^2
$$
and
$$
\phi_2(\alpha) = f(x_k) - \beta \alpha \|\nabla f(x_k)\|^2
$$

Note, that Goldstein-Armijo conditions determine the location of the function $$\phi(\alpha)$$ between $$\phi_1(\alpha)$$ and $$\phi_2(\alpha)$$. Typically, we choose $$\alpha = \rho$$ and $$\beta = 1 - \rho$$, while $$ \rho \in (0.5, 1)$$.

![](../backtracking.svg)

# References

* Numerical Optimization by J.Nocedal and S.J.Wright.
