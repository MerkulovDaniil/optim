---
layout: default
title: General optimization problems
parent: Exercises
nav_order: 10
---

# General optimization problems

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & Ax = b
	\end{split}
	$$

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & a^\top x ≤ b,
	\end{split}
	$$

	where $$a \neq 0$$

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & l \preceq x \preceq u,
	\end{split}
	$$

	where $$l \preceq u$$

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & 1^\top x = 1, \\
	& x \succeq 0 
	\end{split}
	$$

	This problem can be considered as a simplest portfolio optimization problem.

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & 1^\top x = \alpha, \\
	& 0 \preceq x \preceq 1,
	\end{split}
	$$

	where $$\alpha$$ is an integer between $$0$$ and $$n$$. What happens if $$\alpha$$ is not an integer (but
satisfies $$0 \leq \alpha \leq n$$)? What if we change the equality to an inequality $$1^\top x \leq \alpha$$?

1. Give an explicit solution of the following QP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & x^\top A x \leq 1,
	\end{split}
	$$

	where $$A \in \mathbb{S}^n_{++}, c \neq 0$$. What is the solution if the problem is not convex $$(A \notin \mathbb{S}^n_{++})$$ (Hint: consider eigendecomposition of the matrix: $$A = Q \mathbf{diag}(\lambda)Q^\top = \sum\limits_{i=1}^n \lambda_i q_i q_i^\top$$) and different cases of $$\lambda >0, \lambda=0, \lambda<0$$?

1. Give an explicit solution of the following QP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & (x - x_c)^\top A (x - x_c) \leq 1,
	\end{split}
	$$

	where $$A \in \mathbb{S}^n_{++}, c \neq 0, x_c \in \mathbb{R}^n$$.

1. Give an explicit solution of the following QP.
	
	$$
	\begin{split}
	& x^\top Bx \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & x^\top A x \leq 1,
	\end{split}
	$$

	where $$A \in \mathbb{S}^n_{++}, B \in \mathbb{S}^n_{+}$$.

1.  Consider the equality constrained least-squares problem
	
	$$
	\begin{split}
	& \|Ax - b\|_2^2 \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & Cx = d,
	\end{split}
	$$

	where $$A \in \mathbb{R}^{m \times n}$$ with $$\mathbf{rank }A = n$$, and $$C \in \mathbb{C}^{k \times n}$$ with $$\mathbf{rank }C = k$$. Give the KKT conditions, and derive expressions for the primal solution $$x^*$$ and the dual solution $$\lambda^*$$.

1. Derive the KKT conditions for the problem
	
	$$
	\begin{split}
	& \mathbf{tr \;}X - \log\text{det }X \to \min\limits_{X \in \mathbb{S}^n_{++} }\\
	\text{s.t. } & Xs = y,
	\end{split}
	$$

	where $$y \in \mathbb{R}^n$$ and $$s \in \mathbb{R}^n$$ are given with $$y^\top s = 1$$. Verify that the optimal solution is given by

	$$
	X^* = I + yy^\top - \dfrac{1}{s^\top s}ss^\top
	$$

1.  Supporting hyperplane interpretation of KKT conditions. Consider a **convex** problem with no equality constraints
	
	$$
	\begin{split}
	& f_0(x) \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & f_i(x) \leq 0, \quad i = [1,m]
	\end{split}
	$$

	Assume, that $$\exists x^* \in \mathbb{R}^n, \mu^* \in \mathbb{R}^m$$ satisfy the KKT conditions
	
	$$
	\begin{split}
    & \nabla_x L (x^*, \mu^*) = \nabla f_0(x^*) + \sum\limits_{i=1}^m\mu_i^*\nabla f_i(x^*) = 0 \\
    & \mu^*_i \geq 0, \quad i = [1,m] \\
    & \mu^*_i f_i(x^*) = 0, \quad i = [1,m]\\
    & f_i(x^*) \leq 0, \quad i = [1,m]
	\end{split}
	$$

	Show that

	$$
	\nabla f_0(x^*)^\top (x - x^*) \geq 0
	$$

	for all feasible $$x$$. In other words the KKT conditions imply the simple optimality criterion or $$\nabla f_0(x^*)$$ defines a supporting hyperplane to the feasible set at $$x^*$$.
