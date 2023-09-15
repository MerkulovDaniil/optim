---
layout: default
parent: Theory
title: Conjugate function
nav_order: 5
---

# Conjugate (dual) function

Let $$f: \mathbb{R}^n \to \mathbb{R}$$. 
The function $$f^*: \mathbb{R}^n \to \mathbb{R}$$ is called convex conjugate (Fenchel's conjugate, dual, Legendre transform) $$f(x)$$ and is defined as follows:

$$
f^*(y) = \sup\limits_{x \in \mathbf{dom} \; f} \left( \langle y,x\rangle - f(x)\right).
$$

Let's notice, that the domain of the function $$f^*$$  is the set of those $$y$$, where the supremum is finite. 
![](../conj.svg)

## Properties
* $$f^*(y)$$ - is always a closed convex function (a point-wise supremum of closed convex functions) on $$y$$.
(Function $$f:X\rightarrow R$$ is called closed if $$\mathbf{epi}(f)$$ is a closed set in $$X\times R$$.)
* Fenchel–Young inequality: 
	
	$$
	f(x) + f^*(y) \ge \langle y,x \rangle
	$$

* Let the functions $$f(x), f^*(y), f^{**}(x)$$ be defined on the $$\mathbb{R}^n$$. Then $$f^{**}(x) = f(x)$$ if and only if $$f(x)$$ - is a proper convex function (Fenchel - Moreau theorem).
(proper convex function = closed convex function)

* Consequence from Fenchel–Young inequality: $$f(x) \ge f^{**}(x)$$. 

![](../doubl_conj.svg)

* In case of differentiable function, $$f(x)$$ - convex and differentiable, $$\mathbf{dom}\; f = \mathbb{R}^n$$. Then $$x^* = \underset{x}{\operatorname{argmin}} \langle x,y\rangle - f(x)$$. Therefore $$y = \nabla f(x^*)$$. That's why:
	
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

* Let $$f(x) \le g(x)\;\; \forall x \in X$$. Let also $$f^*(y), g^*(y)$$ be defined on $$Y$$. Then $$\forall x \in X, \forall y \in Y$$
	
	$$
	f^*(y) \ge g^*(y) \;\;\;\;\;\; f^{**}(x) \le g^{**}(x)
	$$

## Examples

The scheme of recovering the convex conjugate is pretty algorithmic:
1. Write down the definition $$f^*(y) = \sup\limits_{x \in \mathbf{dom} \; f} \left( \langle y,x\rangle - f(x)\right)  = \sup\limits_{x \in \mathbf{dom} \; g} g(x,y)$$.
1. Find those $$y$$, where $$ \sup\limits_{x \in \mathbf{dom} \; g} g(x,y)$$ is finite. That's the domain of the dual function $$f^*(y)$$.
1. Find $$x^*$$, which maximize $$g(x,y)$$ as a function on $$x$$. $$f^*(y) = g(x^*, y)$$.

{: .example}
>Find $$f^*(y)$$, if $$f(x) = ax + b$$.
><details><summary>Solution</summary>
>1. By definition: 
>$$
>f^*(y) = \sup\limits_{x \in \mathbb{R}} [ yx - f(x) ]=\sup\limits_{x \in \mathbb{R}} g(x,y) \quad \mathbf{dom} \; f^* = \{y \in \mathbb{R} : \sup\limits_{x \in \mathbb{R}} g(x,y) \text{ is finite}\}
>$$
>2. Consider the function whose supremum is the conjugate: 
>$$
>g(x,y) =  yx - f(x) = yx - ax - b = x(y - a) - b.
>$$
>3. Let's determine the domain of the function (i.e. those $y$ for which $\sup$ is finite). This is a single point, $y = a$. Otherwise one may choose such $x$
><br/>
>4. Thus, we have: $\mathbf{dom} \; f^* = \{a\}; f^*(a) = -b$
></details>

{: .example}
>Find $f^*(y)$, if $f(x) = -\log x, \;\; x\in \mathbb{R}_{++}$.
><details><summary>Solution</summary>
>1. Consider the function whose supremum defines the conjugate:
>$$
>g(x,y) = \langle y,x\rangle - f(x) = yx + \log x.
>$$
>2. This function is unbounded above when $y \ge 0$. Therefore, the domain of $f^*$ is $\mathbf{dom} \; f^* = \{y < 0\}$.
><br/>
>3. This function is concave and its maximum is achieved at the point with zero gradient:
>$$
>\dfrac{\partial}{\partial x} (yx + \log x) = \dfrac{1}{x} + y = 0.
>$$
>Thus, we have $x = -\dfrac1y$ and the conjugate function is:
>$$
>f^*(y) = -\log(-y) - 1.
>$$
></details>

{: .example}
>Find $f^*(y)$, if $f(x) = e^x$.
><details><summary>Solution</summary>
>1. Consider the function whose supremum defines the conjugate:
>$$
>g(x,y) = \langle y,x\rangle - f(x) = yx - e^x.
>$$
>2. This function is unbounded above when $y < 0$. Thus, the domain of $f^*$ is $\mathbf{dom} \; f^* = \{y \ge 0\}$.
><br/>
>3. The maximum of this function is achieved when $x = \log y$. Hence:
>$$
>f^*(y) = y \log y - y,
>$$
>assuming $0 \log 0 = 0$.
></details>

{: .example}
>Find $f^*(y)$, if $f(x) = x \log x, x \neq 0,$ and $f(0) = 0, \;\;\; x \in \mathbb{R}_+$.
><details><summary>Solution</summary>
>1. Consider the function whose supremum defines the conjugate:
>$$
>g(x,y) = \langle y,x\rangle - f(x) = xy - x \log x.
>$$
>2. This function is upper bounded for all $y$. Therefore, $\mathbf{dom} \; f^* = \mathbb{R}$.
><br/>
>3. The maximum of this function is achieved when $x = e^{y-1}$. Hence:
>$$
>f^*(y) = e^{y-1}.
>$$
></details>

{: .example}
>Find $f^*(y)$, if $f(x) =\frac{1}{2} x^T A x, \;\;\; A \in \mathbb{S}^n_{++}$.
><details><summary>Solution</summary>
>1. Consider the function whose supremum defines the conjugate:
>$$
>g(x,y) = \langle y,x\rangle - f(x) = y^T x - \frac{1}{2} x^T A x.
>$$
>2. This function is upper bounded for all $y$. Thus, $\mathbf{dom} \; f^* = \mathbb{R}$.
><br/>
>3. The maximum of this function is achieved when $x = A^{-1}y$. Hence:
>$$
>f^*(y) =  \frac{1}{2} y^T A^{-1} y.
>$$
></details>

{: .example}
>Find $f^*(y)$, if $f(x) = \max\limits_{i} x_i, \;\;\; x \in \mathbb{R}^n$.
><details><summary>Solution</summary>
>1. Consider the function whose supremum defines the conjugate:
>$$
>g(x,y) = \langle y,x\rangle - f(x) = y^T x - \max\limits_{i} x_i.
>$$
>2. Observe that if vector $y$ has at least one negative component, this function is not bounded by $x$.
><br/>
>3. If $y \succeq 0$ and $1^T y > 1$, then $y \notin \mathbf{dom} \; f^*(y)$.
><br/>
>4. If $y \succeq 0$ and $1^T y < 1$, then $y \notin \mathbf{dom} \; f^*(y)$.
><br/>
>5. Only left with $y \succeq 0$ and $1^T y = 1$. In this case, $x^T y \le \max\limits_i x_i$.
><br/>
>6. Hence, $f^*(y) = 0$.
></details>