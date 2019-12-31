---
layout: default
parent: Theory
title: Conjugate function
nav_order: 5
---

# Conjugate (dual) function

Let $$f: \mathbb{R}^n \to \mathbb{R}$$. 
The function $$f^*: \mathbb{R}^n \to \mathbb{R}$$ is called convex conjugate (Fenchel's conjugate, dual) $$f(x)$$ and is defined as follows:

$$
f^*(y) = \sup\limits_{x \in \mathbf{dom} \; f} \left( \langle y,x\rangle - f(x)\right).
$$

Let's notice, that the domain of the function $$f^*$$  is the set of those $$y$$, where the supremum is finite. 
![](../conj.svg)

## Properties
* $$f^*(y)$$ - always convex function (point-wise supremum of convex functions) on $$y$$
* Fenchel–Young inequality: 
	
	$$
	f(x) + f^*(y) \ge \langle y,x \rangle
	$$

* Let the functions $$f(x), f^*(y), f^{**}(x)$$ are defined on the $$\mathbb{R}^n$$. Then, $$f^{**}(x) = f(x)$$ if and only if $$f(x)$$ - proper convex function (Fenchel - Moreau theorem).

* Consequence from Fenchel–Young inequality: $$f(x) \ge f^{**}(x)$$ 

![](../doubl_conj.svg)

* The Legendre transformation as a special case of Fenchel's conjugate (in case of differentiable function). Let $$f(x)$$ - convex and differentiable, $$\mathbf{dom}\; f = \mathbb{R}^n$$. Then $$x^* = \underset{x}{\operatorname{argmin}} \langle x,y\rangle - f(x)$$. In that case $$y = \nabla f(x^*)$$. That's why:
	
	$$
	f^*(y) = \langle \nabla f(x^*), x^* \rangle - f(x^*)
	$$

	$$
	f^*(y) = \langle \nabla f(z), z \rangle - f(z), \;\;\;\;\;\; y = \nabla f(z), \;\; z \in \mathbb{R}^n
	$$

* Let $$f(x,y) = f_1(x) + f_2(y)$$, where $$f_1, f_2$$ - convex functions, then 
	
	$$
	f^*(p,q) = f_1^*(p) + f_2^*(q)
	$$

* Let $$f(x) \le g(x)\;\; \forall x \in X$$. Let also $$f^*(y), g^*(y)$$ are defined on $$Y$$. Then $$\forall x \in X, \forall y \in Y$$
	
	$$
	f^*(y) \ge g^*(y) \;\;\;\;\;\; f^{**}(x) \le g^{**}(x)
	$$

## Examples

The scheme of recovering the convex conjugate is pretty algorithmic:
1. Write down the definition $$f^*(y) = \sup\limits_{x \in \mathbf{dom} \; f} \left( \langle y,x\rangle - f(x)\right)  = \sup\limits_{x \in \mathbf{dom} \; f} f(x,y)$$
1. Find those $$y$$, where $$ \sup\limits_{x \in \mathbf{dom} \; f} f(x,y)$$ is finite. That's the domain of the dual function $$f^*(y)$$
1. Find $$x^*$$, which maximize $$f(x,y)$$ as a function on $$x$$. $$f^*(y) = f(x^*, y)$$

### 1 
Find $$f^*(y)$$, if $$f(x) = ax + b$$

Решение:
* Рассмотрим функцию, супремумом которой является сопряженная: $$\langle y,x\rangle - f(x) = yx - ax - b$$
* Построим область определения (т.е. те $$y$$, для которых $$\sup$$ конечен). Это одна точка $$y = a$$
* Значит, $$f^*(a) = -b$$

### 2
Find $$f^*(y)$$, if $$f(x) = -\log x, \;\; x\in \mathbb{R}_{++}$$

Решение:
* Рассмотрим функцию, супремумом которой является сопряженная: $$\langle y,x\rangle - f(x) = yx + \log x$$.
* Эта функция не ограничена сверху при $$y \ge 0$$. Значит, $$\mathbf{dom} \; f^* = \{y < 0\}$$
* Её максимум достигается при $$x = -1/y$$. Значит, $$f^*(y) = -\log(-y) - 1$$

### 3
Find $$f^*(y)$$, if $$f(x) = e^x$$

Решение:
* Рассмотрим функцию, супремумом которой является сопряженная: $$\langle y,x\rangle - f(x) = yx -e^x$$.
* Эта функция не ограничена сверху при $$y < 0$$. Значит, $$\mathbf{dom} \; f^* = \{y \ge 0\}$$ (с нулем лучше поработать аккуратнее)
* Её максимум достигается при $$x = \log y$$. Значит, $$f^*(y) = y \log y - y$$. Полагая, что $$0 \log 0 = 0$$.

### 4
Find $$f^*(y)$$, if $$f(x) = x \log x, x \neq 0, \;\;\; f(0) = 0, \;\;\; x \in \mathbb{R}_+$$

Решение:
* Рассмотрим функцию, супремумом которой является сопряженная: $$\langle y,x\rangle - f(x) =xy - x \log x$$.
* Эта функция ограничена сверху при всех $$y$$. Значит, $$\mathbf{dom} \; f^* = \mathbb{R}$$ (с нулем лучше поработать аккуратнее)
* Её максимум достигается при $$x = e^{y-1}$$. Значит, $$f^*(y) = e^{y-1}$$.

### 5
Find $$f^*(y)$$, if $$f(x) =\frac{1}{2} x^T A x, \;\;\; A \in \mathbb{S}^n_{++}$$

Решение:
* Рассмотрим функцию, супремумом которой является сопряженная: $$\langle y,x\rangle - f(x) =y^Tx - \frac{1}{2}x^TAx$$.
* Эта функция ограничена сверху при всех $$y$$. Значит, $$\mathbf{dom} \; f^* = \mathbb{R}$$ (с нулем лучше поработать аккуратнее)
* Её максимум достигается при $$x = A^{-1}y$$. Значит, $$f^*(y) =  \frac{1}{2}y^TA^{-1}y$$.

### 6
Find $$f^*(y)$$, if $$f(x) =\max\limits_{i} x_i, \;\;\; x \in \mathbb{R}^n$$

Решение:
* Рассмотрим функцию, супремумом которой является сопряженная: $$\langle y,x\rangle - f(x) =y^Tx - \max\limits_{i}x_i$$.
* Заметим, что если вектор $$y$$ имеет хотя бы одну отрицательную компоненту, то эта функция не ограничена по $$x$$.
* Пусть теперь $$y \succeq 0, \;\;\; 1^T y > 1$$. $$y \notin \mathbf{dom \; f^*(y)}$$
* Пусть теперь $$y \succeq 0, \;\;\; 1^T y < 1$$. $$y \notin \mathbf{dom \; f^*(y)}$$
* Остается только $$y \succeq 0, \;\;\; 1^T y = 1$$. Тогда $$x^Ty \le \max\limits_i x_i$$
* Значит, $$f^*(y) = 0$$.
