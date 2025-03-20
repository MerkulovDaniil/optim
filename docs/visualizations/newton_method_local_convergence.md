---
title: "Local convergence of Newton method"
---

It is well known that Newton method converges quadratically to the root of a function if the initial point is close enough to the root and the Hessian is Lipschitz continuous.
$$
x_{k+1} = x_k - \left[\nabla^2 f(x_k)\right]^{-1} \nabla f(x_k)
$$
If you will consider the following function:
$$
f(x) = \begin{cases}
(x - 1)^2, & x \leq -1 \\
-\frac{1}{4}x^4 + \frac{5}{2}x^2 + \frac{7}{4}, & -1 < x < 1 \\
(x + 1)^2, & x \geq 1
\end{cases}
$$
You will see, that it is strongly convex function with smooth gradient (i.e. Lipschitz continuous Hessian).

![Note, that the Hessian is Lipschitz continuous and strictly positive](newton_piecewise_smoooth.pdf)

:::{.video}
newton_method_local_convergence.mp4
:::


[Code](newton_method_local_convergence.py)