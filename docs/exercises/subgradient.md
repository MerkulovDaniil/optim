---
layout: default
title: Subgradient and subdifferential
parent: Exercises
nav_order: 6
---

# Subgradient and subdifferential

1. Prove, that $$x_0$$ - is the minimum point of a convex function $$f(x)$$ if and only if $$0 \in \partial f(x_0)$$
1. Find $$\partial f(x)$$, if $$f(x) = \text{ReLU}(x) = \max \{0, x\}$$
1. Find $$\partial f(x)$$, if $$f(x) = \text{Leaky ReLU}(x) = \begin{cases}
    x & \text{if } x > 0, \\
    0.01x & \text{otherwise}.
\end{cases}$$
1. Find $$\partial f(x)$$, if $$f(x) = \|x\|_p$$ при $$p = 1,2, \infty$$
1. Find $$\partial f(x)$$, if $$f(x) = \|Ax - b\|_1^2$$
1. Find $$\partial f(x)$$, if $$f(x) = e^{\|x\|}$$. Try do the task for an arbitrary norm. At least, try $$\|\cdot\| = \|\cdot\|_{\{2,1,\infty\}}$$.
1. Describe the connection between subgradient of a scalar function $$f: \mathbb{R} \to \mathbb{R}$$ and global linear lower bound, which support (tangent) the graph of the function at a point.
1. What can we say about subdifferential of a convex function in those points, where the function is differentiable?
1. Does the subgradient coincide with the gradient of a function if the function is differentiable? Under which condition it holds?
1. If the function is convex on $$S$$, whether $$\partial f(x) \neq \emptyset  \;\;\; \forall x \in S$$ always holds or not?
1. Find $$\partial f(x)$$, if $$f(x) = x^3$$
1. Find $$f(x) = \lambda_{max} (A(x)) = \sup\limits_{\|y\|_2 = 1} y^T A(x)y$$, где $$A(x) = A_0 + x_1A_1 + \ldots + x_nA_n$$, all the matrices $$A_i \in \mathbb{S}^k$$ are symmetric and defined.
1. Find subdifferential of a function $$f(x,y) = x^2 + xy + y^2 + 3\vert x + y − 2\vert$$ at points $$(1,0)$$ and $$(1,1)$$.
1. Find subdifferential of a function $$f(x) = \sin x$$ on the set $$X = [0, \frac32 \pi]$$.
1. Find subdifferential of a function $$f(x) = \vert c^{\top}x\vert, \; x \in \mathbb{R}^n$$.
1. Find subdifferential of a function $$f(x) = \|x\|_1, \; x \in \mathbb{R}^n$$.
1. Suppose, that if $$f(x) = \|x\|_\infty$$. Prove that
    $$
    \partial f(0) = \textbf{conv}\{\pm e_1, \ldots , \pm e_n\},
    $$
    where $$e_i$$ is $$i$$-th canonical basis vector (column of identity matrix).
