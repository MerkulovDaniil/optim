---
title: Convex functions
order: 5
---

## Convex functions{.unnumbered}

1. Show, that $f(x) = \|x\|$ is convex on $\mathbb{R}^n$.
1. Show, that $f(x) = c^\top x + b$ is convex and concave. 
1. Show, that $f(x) = x^\top Ax$, where $A\succeq 0$ - is convex on $\mathbb{R}^n$.
1. Show, that $f(A) = \lambda_{max}(A)$ - is convex, if $A \in S^n$.
1. Prove, that $-\log\det X$ is convex on $X \in S^n_{++}$.
1. Show, that $f(x)$ is convex, using first and second order criteria, if $f(x) = \sum\limits_{i=1}^n x_i^4$.
1. Find the set of $x \in \mathbb{R}^n$, where the function $f(x) = \dfrac{-1}{2(1 + x^\top x)}$ is convex, strictly convex, strongly convex?  
1. Find the values of $a,b,c$, where $f(x,y,z) = x^2 + 2axy + by^2 + cz^2$ is convex, strictly convex, strongly convex? 
1. Выпуклы ли следующие функции: $f(x) = e^x - 1, \; x \in \mathbb{R};\;\;\; f(x_1, x_2) = x_1x_2, \; x \in \mathbb{R}^2_{++};\;\;\; f(x_1, x_2) = 1/(x_1x_2), \; x \in \mathbb{R}^2_{++}$?
1. Докажите, что множество $S = \left\{ x \in \mathbb{R}^n \mid \prod\limits_{i=1}^n x_i \geq 1 \right\}$ выпукло.
1. Prove, that function $f(X) = \mathbf{tr}(X^{-1}), X \in S^n_{++}$ is convex, while $g(X) = (\det X)^{1/n}, X \in S^n_{++}$ is concave.
1. Kullback–Leibler divergence between $p,q \in \mathbb{R}^n_{++}$ is:
	
	$$
	D(p,q) = \sum\limits_{i=1}^n (p_i \log(p_i/q_i) - p_i + q_i)
	$$
	
	Prove, that $D(p,q) \geq 0 \forall p,q \in \mathbb{R}^n_{++}$ and $D(p,q) = 0 \leftrightarrow p = q$  
	
	Hint: 
	$$
	D(p,q) = f(p) - f(q) - \nabla f(q)^\top (p-q), \;\;\;\; f(p) = \sum\limits_{i=1}^n p_i \log p_i
	$$
1. Let $x$ be a real variable with the values $a_1 < a_2 < \ldots < a_n$ with probabilities $\mathbb{P}(x = a_i) = p_i$. Derive the convexity or concavity of the following functions from $p$ on the set of $\left\{p \mid \sum\limits_{i=1}^n p_i = 1, p_i \ge 0 \right\}$  

	* $\mathbb{E}x$
	* $\mathbb{P}\{x \ge \alpha\}$
	* $\mathbb{P}\{\alpha \le x \le \beta\}$
	* $\sum\limits_{i=1}^n p_i \log p_i​$
	* $\mathbb{V}x = \mathbb{E}(x - \mathbb{E}x)^2$
	* $\mathbf{quartile}(x) = {\operatorname{inf}}\left\{ \beta \mid \mathbb{P}\{x \le \beta\} \ge 0.25 \right\}$ 
1.  Определения выпуклости и сильной выпуклости. Критерии выпуклости и сильной выпуклости первого и второго порядков
1.  Геометрическая интерпретация выпуклости и сильной выпуклости. (подпирание прямой и параболой)
1.  Приведите различные три операции, сохраняющие выпуклость функции.
1.  Доказать, что для $a,b \ge 0; \;\;\; \theta \in [0,1]$

	* $- \log \left( \dfrac{a+b}{2}\right) \le -\dfrac{\log a + \log b}{2}$
	* $a^\theta b^{1-\theta} \le \theta a + (1 - \theta)b$
	*  Hölder’s inequality: $\sum\limits_{i=1}^n x_i y_i \le \left( \sum\limits_{i=1}^n \vert x_i\vert ^p\right)^{1/p} \left( \sum\limits_{i=1}^n \vert y_i\vert^p\right)^{1/p}$. For $p >1, \;\; \dfrac{1}{p} + \dfrac{1}{q} = 1$.

	For $x, y \in \mathbb{R}^n$

1.  Доказать, что матричная норма $f(X) = \|X\|_2 = \sup\limits_{y \in \mathbb{R}^n} \dfrac{\|Xy\|_2}{\|y\|_2}$ выпукла.
1.  Доказать, что:

	* если $f(x)$ - выпукла, то $\exp f(x)$ также выпукла.
	* если $f(x)$ - выпукла, то $g(x)^p$ выпукла для $p \ge 1, f(x) \ge 0$.
	* если $f(x)$ - вогнута, то $1/f(x)$ выпукла для $f(x) > 0$.

1.  Выпукла ли функция $f(X, y) = y^T X^{-1}y$  на множестве $\mathbf{dom} f = \{X, y \mid X + X^T \succeq 0\}$ ? Известно, что эта функция выпукла, если $X$ - симметричная матрица (упражнение - доказать). Докажите выпуклость или приведите простой контрпример.
1.  Пусть функция $h(x)$ - выпуклая на $\mathbb{R}$ неубывающая функция, кроме того: $h(x) = 0$ при $x \le 0$. Докажите, что функция $h\left(\|x\|_2\right)$ выпукла на $\mathbb{R}^n$.
1.  Is the function returning the arithmetic mean of vector coordinates is a convex one: $a(x) = \frac{1}{n}\sum\limits_{i=1}^n x_i$, what about geometric mean: $g(x) = \prod\limits_{i=1}^n \left(x_i \right)^{1/n}$?
1.  Show, that the following function is convex on the set of all positive denominators
	
	$$
	f(x) = \dfrac{1}{x_1 - \dfrac{1}{x_2 - \dfrac{1}{x_3 - \dfrac{1}{\ldots}}}}, x \in \mathbb{R}^n
	$$

1. Влияют ли линейные члены квадратичной функции на ее выпуклость? Сильную выпуклость?
1. Пусть $f(x) : \mathbb{R}^n \to \mathbb{R}$ такова, что $\forall x,y \to f\left( \dfrac{x+y}{2}\right) \leq \dfrac{1}{2}(f(x)+f(y))$. Является ли такая функция выпуклой?
1. Find the set, on which the function $f(x,y) = e^{xy}$ will be convex.
1. Study the following function of two variables $f(x,y) = e^{xy}$.

	1. Is this function convex?
	1. Prove, that this function will be convex on the line $x = y$.
	1. Find another set in $\mathbb{R}^2$, on which this function will be convex.
1. Is $f(x) = -x \ln x - (1-x) \ln (1-x)$ convex?
1. Prove, that adding $\lambda \|x\|_2^2$ to any convex function $g(x)$ ensures strong convexity of a resulting function $f(x) = g(x) + \lambda \|x\|_2^2$. Find the constant of the strong convexity $\mu$.
1. Prove, that function
	
	$$
	f(x) = \log\left( \sum\limits_{i=1}^n e^{x_i}\right)
	$$

	is convex using any differential criterion.
1. Prove, that a function $f$ is strongly convex with parameter $\mu$ if and only if the function
	$$
	x \mapsto f(x)- \frac{\mu}{2} \|x\|^{2}
	$$
	is convex.

1. Give an example of a function, that satisfies Polyak Lojasiewicz condition, but doesn't have convexity property.
1. Prove, that if $g(x)$ - convex function, then $f(x) = g(x) + \dfrac{\lambda}{2}\|x\|^2_2$ will be strongly convex function.
1. Find then $f(x) = x^T A x$ is strongly convex and find strong convexity constant.
1. Let $f: \mathbb{R}^n \to \mathbb{R}$ be the following function:
    $$
    f(x) = \sum\limits_{i=1}^k x_{\lfloor i \rfloor},
    $$
    where $1 \leq k \leq n$, while the symbol $x_{\lfloor i \rfloor}$ stands for the $i$-th component of sorted ($x_{\lfloor 1 \rfloor}$ - maximum component of $x$ and $x_{\lfloor n \rfloor}$ - minimum component of $x$) vector of $x$. Show, that $f$ is a convex function.

1. Consider the function $f(x) = x^d$, where $x \in \mathbb{R}_{+}$. Fill the following table with ✅ or ❎. Explain your answers

	| $d$ | Convex | Concave | Strictly Convex | $\mu$-strongly convex |
	|:-:|:-:|:-:|:-:|:-:|
	| $-2, x \in \mathbb{R}_{++}$| | | | |
	| $-1, x \in \mathbb{R}_{++}$| | | | |
	| $0$| | | | |
	| $0.5$ | | | | |
	|$1$ | | | | |
	| $\in (1; 2)$ | | | | |
	| $2$| | | | |
	| $> 2$| | | |  

1. Prove that the entropy function, defined as

	$$
	f(x) = -\sum_{i=1}^n x_i \log(x_i),
	$$

	with $\text{dom}(f) = \{x \in \R^n_{++} : \sum_{i=1}^n x_i = 1\}$, is strictly concave.  

1. Show, that the function $f: \mathbb{R}^n_{++} \to \mathbb{R}$ is convex if $f(x) = - \prod\limits_{i=1}^n x_i^{\alpha_i}$ if $\mathbf{1}^T \alpha = 1, \alpha \succeq 0$.

1. Show that the maximum of a convex function $f$ over the polyhedron $P = \text{conv}\{v_1, \ldots, v_k\}$ is achieved at one of its vertices, i.e.,

	$$
	\sup_{x \in P} f(x) = \max_{i=1, \ldots, k} f(v_i).
	$$

	A stronger statement is: the maximum of a convex function over a closed bounded convex set is achieved at an extreme point, i.e., a point in the set that is not a convex combination of any other points in the set. (you do not have to prove it). *Hint:* Assume the statement is false, and use Jensen’s inequality.

1. Show, that the two definitions of $\mu$-strongly convex functions are equivalent:
	1. $f(x)$ is $\mu$-strongly convex $\iff$ for any $x_1, x_2 \in S$ and $0 \le \lambda \le 1$ for some $\mu > 0$:
		
		$$
		f(\lambda x_1 + (1 - \lambda)x_2) \le \lambda f(x_1) + (1 - \lambda)f(x_2) - \mu \lambda (1 - \lambda)\|x_1 - x_2\|^2
		$$

	1. $f(x)$ is $\mu$-strongly convex $\iff$ if there exists $\mu>0$ such that the function $f(x) - \dfrac{\mu}{2}\Vert x\Vert^2$ is convex.