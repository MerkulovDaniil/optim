---
title: Dual norm
order: 6
---

# Dual norm

![p-norm and q-norm are dual if this holds](dual_pq.svg)

Let $\Vert x\Vert$ be the norm in the primal space $x \in S \subseteq \mathbb{R}^n$, then the following expression defines dual norm:

$$
\Vert y\Vert _\star = \sup\limits_{\Vert x\Vert  \leq 1} \langle y,x\rangle
$$

The intuition for the finite-dimensional space is how the linear function (element of the dual space) $f_y(\cdot)$ could stretch the elements of the primal space with respect to their size, i.e. $\Vert y\Vert _* = \sup\limits_{x \neq 0} \dfrac{\langle y,x\rangle}{\Vert x\Vert }$.

# Properties
* One can easily define the dual norm as:
	
	$$
	\Vert x\Vert _* = \sup\limits_{y \neq 0} \dfrac{\langle y,x\rangle}{\Vert y\Vert }
	$$

* The dual norm is also a norm itself
* For any $x \in E, y \in E^*$: $x^\top y \leq \Vert x\Vert  \cdot \Vert y\Vert _*$
* $\left(\Vert x\Vert _p\right)_* = \Vert x\Vert _q$ if $\dfrac{1}{p} + \dfrac{1}{q} = 1$, where $p, q \geq 1$

::: {.callout-example}
The Euclidian norm is self dual $\left(\Vert x\Vert_2\right)_\star = \Vert x\Vert _2$.
:::

:::{.plotly}
dual_balls.html
:::

# Examples

::: {.callout-example}
Let $f(x) = \Vert x\Vert$. Prove that $f^\star(y) = \mathbb{O}_{\Vert y\Vert _\star \leq 1}$

::: {.callout-solution collapse="true"}
<br/>
<br/>
<br/>
<br/>
:::
:::