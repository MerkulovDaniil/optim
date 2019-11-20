---
layout: default
parent: Theory
title: Convex optimization problem
nav_order: 9
---

# Convex optimization problem
Note, that there is an agreement in notation of mathematical programming. The problems of the following type are called **Convex optimization problem**:

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n}\\
\text{s.t. } & g_i(x) \leq 0, \; i = 1,\ldots,m\\
& Ax = b,
\end{split}
\tag{COP}
$$

where all the functions $$f(x), g_1(x), \ldots, g_m(x)$$ are convex and all equality constraints are affine. It sounds a bit strange, but not all convex problems are convex optimization problems. 

$$
\tag{CP}
f(x) \to \min\limits_{x \in S},
$$

Where $$f(x)$$ is convex function, defined on the convex set $$S$$. The neccessity of affine equality constraint is essential see Slater's condition in {% include link.html title = 'Duality' %}. 

For example, this problem is not convex optimization problem (but implies minimizing convex function over the convex set):

$$
\begin{split}
& x_1^2 + x_2^2 \to \min\limits_{x \in \mathbb{R}^n}\\
\text{s.t. } & \dfrac{x_1}{1 + x_2^2} \leq 0\\
& (x_1 + x_2)^2 = 0,
\end{split}
\tag{CP}
$$

while the following equivalent problem is convex optimization problem

$$
\begin{split}
& x_1^2 + x_2^2 \to \min\limits_{x \in \mathbb{R}^n}\\
\text{s.t. } & \dfrac{x_1}{1 + x_2^2} \leq 0\\
& x_1 + x_2 = 0,
\end{split}
\tag{COP}
$$

Such confusion in notation is sometimes being avoided by naming problems of type $$\text{(CP)}$$ as *abstract form convex optimization problem*

# Materials

* [Convex Optimization — Boyd & Vandenberghe @ Stanford](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)