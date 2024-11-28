# Projection


## Projection

1.  Let us have two different points $a, b \in \mathbb{R}^n$. Prove that
    the set of points which in the Euclidean norm are closer to the
    point $a$ than to $b$ make up a half-space. Is this true for another
    norm?

2.  Find $\pi_S (y) = \pi$ if
    $S = \{x \in \mathbb{R}^n \mid \|x - x_c\| \le R \}$, $y \notin S$

3.  Find $\pi_S (y) = \pi$ if
    $S = \{x \in \mathbb{R}^n \mid c^T x = b \}$, $y \notin S$

4.  Find $\pi_S (y) = \pi$ if
    $S = \{x \in \mathbb{R}^n \mid Ax = b, A \in \mathbb{R}^{m \times n}, b \in \mathbb{R}^{m} \}$,
    $y \notin S$

5.  Illustrate the geometric inequality that connects
    $\pi_S(y), y \notin S, x \in S$, from which it follows that
    $\pi_S(y)$ is a projection of the $y$ point onto a convex set of
    $S$.

6.  For which sets does the projection of the point outside this set
    exist? Unique?

7.  Find $\pi_S (y) = \pi$ if
    $S = \{x \in \mathbb{R}^n \mid c^T x \ge b \}$

8.  Find $\pi_S (y) = \pi$ if
    $S = \{x \in \mathbb{R}^n \mid x = x_0 + X \alpha, X \in \mathbb{R}^{n \times m}, \alpha \in \mathbb{R}^{m}\}$,
    $y \in S$

9.  Let $S \subseteq \mathbb{R}^n$ be a closed set, and
    $x \in \mathbb{R}^n$ be a point not lying in it. Show that the
    projection in $l_2$ norm will be unique, while in $l_\infty$ norm
    this statement is not valid.

10. Find the projection of the matrix $X$ on a set of matrices of rank
    $k, \;\;\; X \in \mathbb{R}^{m \times n}, k \leq n \leq m$. In
    Frobenius norm and spectral norm.

11. Find a projection of the $X$ matrix on a set of symmetrical positive
    semi-definite matrices of $X \in \mathbb{R}^{n \times n}$. In
    Frobenius norm and the scalar product associated with it.

12. Find the projection $\pi_S(y)$ of point $y$ onto the set
    $S = \{x_1, x_2 \in \mathbb{R}^2 \mid \mid \vert x_1\vert + \vert x_2\vert = 1 \}$
    in $\| \cdot \|_1$ norm. Consider the different positions of $y$.

13. Find $\pi_S (y) = \pi$, if
    $S = \{x \in \mathbb{R}^n \mid \alpha_i \le x_i \le \beta_i, i = 1, \ldots, n \}$.

14. Prove that projection is a nonexpansive operator, i.e. prove, that
    if $S \in \mathbb{R}^{n}$ is nonempty, closed and convex set, then
    for any $(x_{1}, x_{2}) \in \mathbb{R}^{n} \times \mathbb{R}^{n}$

    $$
     \lVert \pi_{S}(x_{2}) - \pi_{S}(x_{1}) \rVert_{2} \leq \lVert x_{2} - x_{1} \rVert_{2}
     $$
