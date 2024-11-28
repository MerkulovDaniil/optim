# Subgradient and subdifferential


## Subgradient and subdifferential

1.  Prove, that $x_0$ - is the minimum point of a convex function $f(x)$
    if and only if $0 \in \partial f(x_0)$
2.  Find $\partial f(x)$, if $f(x) = \text{ReLU}(x) = \max \{0, x\}$
3.  Find $\partial f(x)$, if
    $f(x) = \text{Leaky ReLU}(x) = \begin{cases}
     x & \text{if } x > 0, \\
     0.01x & \text{otherwise}.
    \end{cases}$
4.  Find $\partial f(x)$, if $f(x) = \|x\|_p$ при $p = 1,2, \infty$
5.  Find $\partial f(x)$, if $f(x) = \|Ax - b\|_1^2$
6.  Find $\partial f(x)$, if $f(x) = e^{\|x\|}$. Try do the task for an
    arbitrary norm. At least, try
    $\|\cdot\| = \|\cdot\|_{\{2,1,\infty\}}$.
7.  Describe the connection between subgradient of a scalar function
    $f: \mathbb{R} \to \mathbb{R}$ and global linear lower bound, which
    support (tangent) the graph of the function at a point.
8.  What can we say about subdifferential of a convex function in those
    points, where the function is differentiable?
9.  Does the subgradient coincide with the gradient of a function if the
    function is differentiable? Under which condition it holds?
10. If the function is convex on $S$, whether
    $\partial f(x) \neq \emptyset  \;\;\; \forall x \in S$ always holds
    or not?
11. Find $\partial f(x)$, if $f(x) = x^3$
12. Find
    $f(x) = \lambda_{max} (A(x)) = \sup\limits_{\|y\|_2 = 1} y^T A(x)y$,
    где $A(x) = A_0 + x_1A_1 + \ldots + x_nA_n$, all the matrices
    $A_i \in \mathbb{S}^k$ are symmetric and defined.
13. Find subdifferential of a function
    $f(x,y) = x^2 + xy + y^2 + 3\vert x + y − 2\vert$ at points $(1,0)$
    and $(1,1)$.
14. Find subdifferential of a function $f(x) = \sin x$ on the set
    $X = [0, \frac32 \pi]$.
15. Find subdifferential of a function
    $f(x) = \vert c^{\top}x\vert, \; x \in \mathbb{R}^n$.
16. Find subdifferential of a function
    $f(x) = \|x\|_1, \; x \in \mathbb{R}^n$.
17. Suppose, that if $f(x) = \|x\|_\infty$. Prove that $$
     \partial f(0) = \textbf{conv}\{\pm e_1, \ldots , \pm e_n\},
     $$ where $e_i$ is $i$-th canonical basis vector (column of identity
    matrix).
