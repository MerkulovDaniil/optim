---
title: Dual norm
order: 6
---

## Dual norm

![p-norm and q-norm are dual if this holds](dual_pq.svg)

Let $\Vert x\Vert$ be the norm in the primal space $x \in S \subseteq \mathbb{R}^n$, then the following expression defines dual norm:

$$
\Vert y\Vert _\star = \sup\limits_{\Vert x\Vert  \leq 1} \langle y,x\rangle
$$

The intuition for the finite-dimensional space is how the linear function (element of the dual space) $f_y(\cdot)$ could stretch the elements of the primal space with respect to their size, i.e. $\Vert y\Vert _* = \sup\limits_{x \neq 0} \dfrac{\langle y,x\rangle}{\Vert x\Vert }$.

## Properties
* One can easily define the dual norm as:
	
	$$
	\Vert y\Vert _* = \sup\limits_{x \neq 0} \dfrac{\langle y,x\rangle}{\Vert x\Vert }
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

## Examples

::: {.callout-example}
Let $f(x) = \Vert x\Vert$. Prove that $f^\star(y) = \mathbb{O}_{\Vert y\Vert _\star \leq 1}$

::: {.callout-solution collapse="true"}
1. By definition of the conjugate function:

	$$
	f^*(y) = \sup_{x} \{ \langle y, x \rangle - f(x) \} = \sup_{x} \{ \langle y, x \rangle - \|x\| \}
	$$

2. Consider the case $\|y\|_* > 1$. By definition of the dual norm, 

	$$
	\Vert y\Vert _* = \sup\limits_{\Vert x\Vert  \leq 1} \langle y,x\rangle > 1
	$$

	Which means, that there is some $x^\dagger$, such that $\|x^\dagger\|\leq 1$, but $\langle y,x^\dagger\rangle > 1$. Now consider the vector $\bar{x} = tx^\dagger$, where $t \in \mathbb{R}^+$. The value of the conjugate function is a supremum, therefore we have the following relation:

	$$
	\begin{split}
	f^*(y) &\geq \langle y, \bar{x} \rangle - \|\bar{x}\| = \langle y, tx^\dagger \rangle - t\|x^\dagger\|\\
	&= t(\langle y, x^\dagger \rangle - \|x^\dagger\|) \to \infty \text{ with } t \to \infty
	\end{split} 
	$$

	Thus, $\|y\|_* > 1$ does not belong to the $\text{dom } f^*$.

1. Consider the case $\|y\|_* \leq 1$. By CBS inequality:

	$$
	\langle y, x \rangle \leq \| y \|_* \| x \| \leq \| x \|
	$$

	Equality holds when $x=0$. Therefore

	$$
	f^*(y) = \sup_{x} \{ \langle y, x \rangle - \|x\| \} = 0
	$$

:::
:::