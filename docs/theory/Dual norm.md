---
layout: default
parent: Theory
title: Dual norm
nav_order: 6
---

# Dual norm

Let $$\|x\|$$ be the norm in the primal space $$x \in S \subseteq \mathbb{R}^n$$, then the following expression defines dual norm:

$$
\|x\|_* = \sup\limits_{\|y\| \leq 1} \langle y,x\rangle
$$

The intuition for the finite-dimensional space is how the linear function (element of the dual space) $$f_y(\cdot)$$ could stretch the elements of the primal space with respect to their size, i.e. $$\|y\|_* = \sup\limits_{x \neq 0} \dfrac{\langle y,x\rangle}{\|x\|}$$

# Properties
* One can easily define the dual norm as:
	
	$$
	\|x\|_* = \sup\limits_{y \neq 0} \dfrac{\langle y,x\rangle}{\|y\|}
	$$

* The dual norm is also a norm itself
* For any $$x \in E, y \in E^*$$: $$x^\top y \leq \|x\| \cdot \|y\|_*$$
* $$\left(\|x\|_p\right)_* = \|x\|_q$$ if $$\dfrac{1}{p} + \dfrac{1}{q} = 1$$, where $$p, q \geq 1$$

# Examples

* Let $$f(x) = \|x\|$$, then $$f^*(y) = \mathbb{O}_{\|y\|_* \leq 1}$$
* The Euclidian norm is self dual $$\left(\|x\|_2\right)_* = \|x\|_2$$.
