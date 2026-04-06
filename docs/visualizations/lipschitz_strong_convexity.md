---
title: "Strong convexity and smoothness bounds"
---

When a function is both $\mu$-strongly convex and $L$-smooth, its Hessian satisfies:

$$
\mu I \preceq \nabla^2 f(x) \preceq L I
$$

This gives two-sided quadratic bounds at every point:

$$
f(x) + \langle \nabla f(x), y - x \rangle + \frac{\mu}{2}\|y-x\|^2 \leq f(y) \leq f(x) + \langle \nabla f(x), y - x \rangle + \frac{L}{2}\|y-x\|^2
$$

The ratio $\kappa = L / \mu$ is the condition number, which determines convergence rates of optimization methods. The animation shows how the function is sandwiched between a lower parabola (strong convexity) and an upper parabola (smoothness) at each point.

![Strong convexity and smoothness](lipschitz_strong_convexity.svg)

:::{.video}
lipschitz_strong_convexity.mp4
:::
