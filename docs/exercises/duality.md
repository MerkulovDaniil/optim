---
layout: default
title: Duality
parent: Exercises
nav_order: 10
---

# Duality

1. **Least Squares.** Let we have the primal problem:
	
	$$
	\begin{split}
	& x^\top x \to \min\limits_{x \in \mathbb{R}^{n} }\\
	\text{s.t. } & Ax = b
	\end{split}
	$$

	1. Find Lagrangian of the primal problem
	1. Find the dual function
	1. Write down the dual problem
	1. Check whether problem holds strong duality or not
	1. Write down the solution of the dual problem
1. **Standard form LP.** Let we have the primal problem:
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^{n} }\\
	\text{s.t. } & Ax = b \\
	& x \succeq 0
	\end{split}
	$$

	1. Find Lagrangian of the primal problem
	1. Find the dual function
	1. Write down the dual problem
	1. Check whether problem holds strong duality or not
	1. Write down the solution of the dual problem
1. **Two-way partitioning problem.** Let we have the primal problem:
	
	$$
	\begin{split}
	& x^\top W x \to \min\limits_{x \in \mathbb{R}^{n} }\\
	\text{s.t. } & x_1^2 = 1, i = 1, \ldots, n \\
	\end{split}
	$$

	1. Find Lagrangian of the primal problem
	1. Find the dual function
	1. Write down the dual problem
	1. Check whether problem holds strong duality or not
	1. Write down the solution of the dual problem
	1. Can you reduce this problem to the eigenvalue problem? 🐱
1. **Entropy maximization.** Let we have the primal problem:
	
	$$
	\begin{split}
	& \sum_i x_i \ln x_i \to \min\limits_{x \in \mathbb{R}^{n} }\\
	\text{s.t. } & Ax \preceq b \\
	& 1^\top x = 1 \\
	& x > 0
	\end{split}
	$$

	1. Find Lagrangian of the primal problem
	1. Find the dual function
	1. Write down the dual problem
	1. Check whether problem holds strong duality or not
	1. Write down the solution of the dual problem
1. **Minimum volume covering ellipsoid.** Let we have the primal problem:
	
	$$
	\begin{split}
	& \ln \text{det} X^{-1} \to \min\limits_{X \in \mathbb{S}^{n}_{++} }\\
	\text{s.t. } & a_i^\top X a_i \leq 1 , i = 1, \ldots, m
	\end{split}
	$$

	1. Find Lagrangian of the primal problem
	1. Find the dual function
	1. Write down the dual problem
	1. Check whether problem holds strong duality or not
	1. Write down the solution of the dual problem
1. **Equality constrained norm minimization.** Let we have the primal problem:
	
	$$
	\begin{split}
	& \|x\| \to \min\limits_{x \in \mathbb{R}^{n} }\\
	\text{s.t. } & Ax = b
	\end{split}
	$$

	1. Find Lagrangian of the primal problem
	1. Find the dual function
	1. Write down the dual problem
	1. Check whether problem holds strong duality or not
	1. Write down the solution of the dual problem
1. **Inequality form LP.** Let we have the primal problem:
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^{n} }\\
	\text{s.t. } & Ax \preeq b \\
	& x \succeq 0
	\end{split}
	$$

	1. Find Lagrangian of the primal problem
	1. Find the dual function
	1. Write down the dual problem
	1. Check whether problem holds strong duality or not
	1. Write down the solution of the dual problem