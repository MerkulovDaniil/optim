---
layout: default
title: Adaptive metric methods
parent: Methods
has_children: True
nav_order: 3
---

It is known, that antigradient $$-\nabla f (x_0)$$ is the direction of the steepest descent of the function $$f(x)$$ at point $$x_0$$. However, we can introduce another concept for choosing the best direction of function decreasing. 

Given $$f(x)$$ and a point $$x_0$$. Define $$B_\varepsilon(x_0) = \{x \in \mathbb{R}^n : d(x, x_0) = \varepsilon^2 \}$$ as the set of points with distance $$\varepsilon$$ to $$x_0$$. Here we presume the existence of a distance function $$d(x, x_0)$$.

$$
x^* = \text{arg}\min_{x \in B_\varepsilon(x_0)} f(x)
$$

Than, we can define another *steepest descent* direction in terms of minimizer of  function on a sphere:

$$
s = \lim_{\varepsilon \to 0} \frac{x^* - x_0}{\varepsilon}
$$

Let us assume that the distance is defined locally by some metric $$A$$:

$$
d(x, x_0) = (x-x_0)^\top A (x-x_0)
$$

Let us also consider first order Taylor approximation of a function $$f(x)$$ near the point $$x_0$$:

$$
\tag{A1}
f(x_0 + \delta x) \approx f(x_0) + \nabla f(x_0)^\top \delta x
$$

Now we can explicitly pose a problem of finding $$s$$, as it was stated above.

$$
\begin{split}
&\min_{\delta x \in \mathbb{R^n}} f(x_0 + \delta x) \\
\text{s.t.}\;& \delta x^\top A \delta x = \varepsilon^2
\end{split}
$$

Using $$\text{(A1)}$$ it can be written as:

$$
\begin{split}
&\min_{\delta x \in \mathbb{R^n}} \nabla f(x_0)^\top \delta x \\
\text{s.t.}\;& \delta x^\top A \delta x = \varepsilon^2
\end{split}
$$

Using Lagrange multipliers method, we can easily conclude, that the answer is:

$$
\delta x = - \frac{2 \varepsilon^2}{\nabla f (x_0)^\top A^{-1} \nabla f (x_0)} A^{-1} \nabla f
$$

Which means, that new direction of steepest descent is nothing else, but $$A^{-1} \nabla f(x_0)$$.

Indeed, if the space is isotropic and $$A = I$$, we immediately have gradient descent formula, while Newton method uses local Hessian as a metric matrix. 
