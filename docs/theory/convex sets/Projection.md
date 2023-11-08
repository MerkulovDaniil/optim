---
title: Projection
order: 5
---

# Definitions
## Distance between point and set
The distance $d$ from point $\mathbf{y} \in \mathbb{R}^n$ to closed set $S \subset \mathbb{R}^n$:

$$
d(\mathbf{y}, S, \| \cdot \|) = \inf\{\|x - y\| \mid x \in S \}
$$


## Projection of a point on set
Projection of a point $\mathbf{y} \in \mathbb{R}^n$ on set $S \subseteq \mathbb{R}^n$ is a point $\pi_S(\mathbf{y}) \in S$: 

$$
\| \pi_S(\mathbf{y}) - \mathbf{y}\| \le \|\mathbf{x} - \mathbf{y}\|, \forall \mathbf{x} \in S
$$

* if a set is open, and a point is beyond this set, then its projection on this set does not exist.
* if a point is in set, then its projection is the point itself
*   $$
    \pi_S(\mathbf{y}) = \underset{\mathbf{x}}{\operatorname{argmin}} \|\mathbf{x}-\mathbf{y}\|
    $$

* Let $S \subseteq \mathbb{R}^n$ - convex closed set. Let the point $\mathbf{y} \in \mathbb{R}^n$ и $\mathbf{\pi} \in S$. Then if for all  $\mathbf{x} \in S$ the inequality holds:
    
    $$
    \langle \pi  -\mathbf{y}, \mathbf{x} - \pi\rangle \ge 0, 
    $$

    then $\pi$ is the projection of the point $\mathbf{y}$ on $S$, so $\pi_S (\mathbf{y}) = \pi$. 
* Let $S \subseteq \mathbb{R}^n$ - affine set. Let we have points $\mathbf{y} \in \mathbb{R}^n$ and $\mathbf{\pi} \in S$. Then $\pi$ is a projection of point $\mathbf{y}$ on $S$, so $\pi_S (\mathbf{y}) = \pi$ if and only if for all $\mathbf{x} \in S$ the inequality holds: 

$$
\langle \pi  -\mathbf{y}, \mathbf{x} - \pi\rangle = 0 
$$

* **Sufficient conditions of existence of a projection**. If $S \subseteq \mathbb{R}^n$ - closed set, then the projection on set $S$ exists for any point.
* **Sufficient conditions of uniqueness of a projection**. If $S \subseteq \mathbb{R}^n$ - closed convex set, then the projection on set $S$ is unique for any point.
 
::: {.callout-example}
Find $\pi_S (y) = \pi$, if $S = \{x \in \mathbb{R}^n \mid \|x - x_0\| \le R \}$, $y \notin S$ 

::: {.callout-solution collapse="true"}

![Projection of point to the ball](proj_cir.gif){#fig-projection_ball}

* Build a hypothesis from the figure: $\pi = x_0 + R \cdot \frac{y - x_0}{\|y - x_0\|}$ 

* Check the inequality for a convex closed set: $(\pi - y)^T(x - \pi) \ge 0$ 

    $$
    \left( x_0 - y + R \frac{y - x_0}{\|y - x_0\|} \right)^T\left( x - x_0 - R \frac{y - x_0}{\|y - x_0\|} \right) =
    $$

    $$
    \left( \frac{(y - x_0)(R - \|y - x_0\|)}{\|y - x_0\|} \right)^T\left( \frac{(x-x_0)\|y-x_0\|-R(y - x_0)}{\|y - x_0\|} \right) =
    $$

    $$
    \frac{R - \|y - x_0\|}{\|y - x_0\|^2} \left(y - x_0 \right)^T\left( \left(x-x_0\right)\|y-x_0\|-R\left(y - x_0\right) \right) = 
    $$

    $$
    \frac{R - \|y - x_0\|}{\|y - x_0\|} \left( \left(y - x_0 \right)^T\left( x-x_0\right)-R\|y - x_0\| \right) =
    $$

    $$
    \left(R - \|y - x_0\| \right) \left( \frac{(y - x_0 )^T( x-x_0)}{\|y - x_0\|}-R \right)
    $$

* The first factor is negative for point selection $y$. The second factor is also negative, which follows from the Cauchy-Bunyakovsky inequality: 

    $$
    (y - x_0 )^T( x-x_0) \le \|y - x_0\|\|x-x_0\|
    $$

    $$
    \frac{(y - x_0 )^T( x-x_0)}{\|y - x_0\|} - R \le \frac{\|y - x_0\|\|x-x_0\|}{\|y - x_0\|} - R = \|x - x_0\| - R \le 0
    $$

:::
:::

::: {.callout-example}
Find $\pi_S (y) = \pi$, if $S = \{x \in \mathbb{R}^n \mid c^T x = b \}$, $y \notin S$. 

::: {.callout-solution collapse="true"}

![Projection of point to the ball](proj_half.gif){#fig-projection_halfspace}

* Build a hypothesis from the figure: $\pi = y + \alpha c$. Coefficient $\alpha$ is chosen so that $\pi \in S$: $c^T \pi = b$, so: 

    $$
    c^T (y + \alpha c) = b
    $$

    $$
    c^Ty + \alpha c^T c = b
    $$

    $$
    c^Ty = b - \alpha c^T c
    $$

* Check the inequality for a convex closed set: $(\pi - y)^T(x - \pi) \ge 0$ 

    $$
    (y + \alpha c - y)^T(x - y - \alpha c) = 
    $$

    $$
     \alpha c^T(x - y - \alpha c) = 
    $$

    $$
     \alpha (c^Tx) - \alpha (c^T y) - \alpha^2 (c^Tc) = 
    $$

    $$
     \alpha b - \alpha (b - \alpha c^T c) - \alpha^2 c^Tc = 
    $$

    $$
     \alpha b - \alpha b + \alpha^2 c^T c - \alpha^2 c^Tc = 0 \ge 0
    $$

:::
:::

::: {.callout-example}
Find $\pi_S (y) = \pi$, if $S = \{x \in \mathbb{R}^n \mid Ax = b, A \in \mathbb{R}^{m \times n},  b \in \mathbb{R}^{m} \}$, $y \notin S$. 

::: {.callout-solution collapse="true"}

![Projection of point to the set of linear equations](proj_poly.gif){#fig-projection_linear_system}

* Build a hypothesis from the figure: $\pi = y + \sum\limits_{i=1}^m\alpha_i A_i = y + A^T \alpha$. Coefficient $\alpha$ is chosen so that $\pi \in S$: $A \pi = b$, so: 

    $$
    A(y + A^T\alpha) = b
    $$

    $$
    Ay = b - A A^T\alpha
    $$

* Check the inequality for a convex closed set: $(\pi - y)^T(x - \pi) \ge 0$ 

    $$
    (y + A^T\alpha  - y)^T(x - y - A^T\alpha) = 
    $$

    $$
     \alpha^T A(x - y - A^T\alpha) = 
    $$

    $$
     \alpha^T (Ax) - \alpha^T (A y) - \alpha^T (AA^T \alpha) = 
    $$

    $$
     \alpha^T b - \alpha^T (b - A A^T\alpha) - \alpha^T AA^T \alpha = 
    $$

    $$
     \alpha^T b - \alpha^T b + \alpha^T AA^T \alpha - \alpha^T AA^T \alpha = 0 \ge 0
    $$ 
:::
:::