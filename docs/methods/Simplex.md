---
layout: default
parent: Methods
title: LP and simplex algorithm
nav_order: 5
---

# What is LP

Generally speaking, all problems with linear objective and linear equalities\inequalities constraints could be considered as Linear Programming. However, there are some widely accepted formulations.

## Standard form
This form seems to be the most intuitive and geometric in terms of visualization. Let us have vectors $$c \in \mathbb{R}^n$$, $$b \in \mathbb{R}^m$$ and matrix $$A \in \mathbb{R}^{m \times n}$$.

$$
\tag{LP.Standard}
\begin{align*}
&\min_{x \in \mathbb{R}^n} c^{\top}x \\
\text{s.t. } & Ax \leq b\\
& x_i \geq 0, \; i = 1,\dots, n
\end{align*}
$$

![](../LP.svg)

## Canonical form

$$
\tag{LP.Canonical}
\begin{align*}
&\min_{x \in \mathbb{R}^n} c^{\top}x \\
\text{s.t. } & Ax = b\\
& x_i \geq 0, \; i = 1,\dots, n
\end{align*}
$$

## Real world problems

### Diet problem
Imagine, that you have to construct a diet plan from some set of products: 🍌🍰🍗🥚🐟. Each of the products has its own vector of nutrients. Thus, all the food information could be processed through the matrix $$W$$. Let also assume, that we have the vector of requirements for each of nutrients $$r \in \mathbb{R}^n$$. We need to find the cheapest configuration of the diet, which meets all the requirements:

$$
\begin{align*}
&\min_{x \in \mathbb{R}^p} c^{\top}x \\
\text{s.t. } & Wx \geq r\\
& x_i \geq 0, \; i = 1,\dots, n
\end{align*}
$$

![](../diet_LP.svg)

### Radiation treatment 



# How to retrieve LP
## Basic transformations
Inequality to equality by increasing the dimension of the problem by $$m$$.

$$
Ax \leq b \leftrightarrow 
\begin{cases}
Ax + z =  b\\
z \geq 0
\end{cases}
$$

unsigned variables to nonnegative variables.

$$
x \leftrightarrow 
\begin{cases}
x = x_+ - x_-\\
x_+ \geq 0 \\
x_- \geq 0
\end{cases}
$$

## Chebyshev approximation problem


$$
\min_{x \in \mathbb{R}^n} \|Ax - b\|_\infty \leftrightarrow \min_{x \in \mathbb{R}^n} \max_{i} |a_i^\top x - b_i|
$$

$$
\begin{align*}
&\min_{t \in \mathbb{R}, x \in \mathbb{R}^n} t \\
\text{s.t. } & a_i^\top x - b_i \leq t, \; i = 1,\dots, n\\
& -a_i^\top x + b_i \leq t, \; i = 1,\dots, n
\end{align*}
$$

## $$l_1$$ approximation problem


$$
\min_{x \in \mathbb{R}^n} \|Ax - b\|_1 \leftrightarrow \min_{x \in \mathbb{R}^n} \sum_{i=1}^n |a_i^\top x - b_i|
$$

$$
\begin{align*}
&\min_{t \in \mathbb{R}^n, x \in \mathbb{R}^n} \mathbf{1}^\top t \\
\text{s.t. } & a_i^\top x - b_i \leq t_i, \; i = 1,\dots, n\\
& -a_i^\top x + b_i \leq t_i, \; i = 1,\dots, n
\end{align*}
$$

# Idea of simplex algorithm

* The Simplex Algorithm walks along the edges of the polytope, at every corner choosing the edge that decreases $$c^\top x$$ most
* This either terminates at a corner, or leads to an unconstrained edge ($$-\infty$$ optimum)

We will illustrate simplex algorithm for the simple inequality form of LP:

$$
\tag{LP.Inequality}
\begin{align*}
&\min_{x \in \mathbb{R}^n} c^{\top}x \\
\text{s.t. } & Ax \leq b
\end{align*}
$$

Definition: a **basis** $$B$$ is a subset of $$n$$ (integer) numbers between $$1$$ and $$m$$, so that $$\text{rank} A_B = n$$. Note, that we can associate submatrix $$A_B$$ and corresponding right-hand side $$b_B$$ with the basis $$B$$. Also, we can derive a point of intersection of all these hyperplanes from basis: $$x_B = A^{-1}_B b_B$$. 

If $$A x_B \leq b$$, then basis $$B$$ is **feasible**. 

A basis $$B$$ is optimal if $$x_B$$ is an optimum of the $$\text{LP.Inequality}$$.

![](../LP_1.svg)

Since we have a basis, we can decompose our objective vector $$c$$ in this basis and find the scalar coefficients $$\lambda_B$$:

$$
\lambda^\top_B A_B = c^\top \leftrightarrow \lambda^\top_B = c^\top A_B^{-1}
$$

## Main lemma

If all components of $$\lambda_B$$ are non-positive and $$B$$ is feasible, then $$B$$ is optimal.

**Proof:**

$$
\begin{align*}
\exists x^*: Ax^* &\leq b, c^\top x^* < c^\top x_B \\
A_B x^* &\leq b_B \\
\lambda_B^\top A_B x^* &\geq \lambda_B^\top b_B \\
c^\top x^* & \geq \lambda_B^\top A_B x_B \\
c^\top x^* & \geq c^\top  x_B \\
\end{align*}
$$

## Changing basis

Suppose, some of the coefficients of $$\lambda_B$$ are positive. Then we need to go through the edge of the polytope to the new vertex (i.e., switch the basis)

![](../LP_2.svg)

$$
x_{B'} = x_B + \mu d = A^{-1}_{B'} b_{B'}
$$

## Finding an initial basic feasible solution

Let us consider $$\text{LP.Canonical}$$.

$$
\begin{align*}
&\min_{x \in \mathbb{R}^n} c^{\top}x \\
\text{s.t. } & Ax = b\\
& x_i \geq 0, \; i = 1,\dots, n
\end{align*}
$$

The proposed algorithm requires an initial basic feasible solution and corresponding basis. To compute this solution and basis, we start by multiplying by $$−1$$ any row $$i$$ of $$Ax = b$$ such that $$b_i < 0$$. This ensures that $$b \geq 0$$. We then introduce artificial variables $$z \in \mathbb{R}^m$$ and consider the following LP:

$$
\tag{LP.Phase 1}
\begin{align*}
&\min_{x \in \mathbb{R}^n, z \in \mathbb{R}^m} 1^{\top}z \\
\text{s.t. } & Ax + Iz = b\\
& x_i, z_j \geq 0, \; i = 1,\dots, n \; j = 1,\dots, m
\end{align*}
$$

which can be written in canonical form $$\min\{\tilde{c}^\top \tilde{x} | \tilde{A}\tilde{x} = \tilde{b}, \tilde{x} \geq 0\}$$ by setting

$$
\tilde{x} = \begin{bmatrix}x\\z\end{bmatrix}, \quad \tilde{A} = [A \; I], \quad \tilde{b} = b, \quad \tilde{c} = \begin{bmatrix}0_n\\1_m\end{bmatrix}
$$

An initial basis for $$\text{LP.Phase 1}$$ is $$\tilde{A}_B = I, \tilde{A}_N = A$$ with corresponding basic feasible solution $$\tilde{x}_N = 0, \tilde{x}_B = \tilde{A}^{-1}_B \tilde{b} = \tilde{b} \geq 0$$. We can therefore run the simplex method on $$\text{LP.Phase 1}$$, which will converge to an optimum $$\tilde{x}^*$$. $$\tilde{x} = (\tilde{x}_N \; \tilde{x}_B)$$. There are several possible outcomes:
* \$$\tilde{c}^\top \tilde{x} > 0$$. Original primal is infeasible.
* \$$\tilde{c}^\top \tilde{x} = 0 \to 1^\top z^* = 0$$. The obtained solution is a start point for the original problem (probably with slight modification).


# About convergence
## [Klee Minty](https://en.wikipedia.org/wiki/Klee%E2%80%93Minty_cube) example

In the following problem simplex algorithm needs to check $$2^n - 1$$ vertexes with $$x_0 = 0$$. 

$$
\begin{align*} & \max_{x \in \mathbb{R}^n} 2^{n-1}x_1 + 2^{n-2}x_2 + \dots + 2x_{n-1} + x_n\\
\text{s.t. } & x_1 \leq 5\\
& 4x_1 + x_2 \leq 25\\
& 8x_1 + 4x_2 + x_3 \leq 125\\
& \ldots\\
& 2^n x_1 + 2^{n-1}x_2 + 2^{n-2}x_3 + \ldots + x_n \leq 5^n\ & x \geq 0 
\end{align*}
$$

[](../LP_KM.svg)

# Summary
* A wide variety of applications could be formulated as the linear programming.
* Simplex algorithm is simple, but could work exponentially long.
* Khachiyan’s ellipsoid method is the first to be proved running at polynomial complexity for LPs. However, it is usually slower than simplex in real problems.
* Interior point methods are the last word in this area. However, good implementations of simplex-based methods and interior point methods are similar for routine applications of linear programming.

# Code

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/LP.ipynb)

# Materials

* [Linear Programming.](https://yadi.sk/i/uhmarI88kCRfw) in V. Lempitsky optimization course.
* [Simplex method.](https://yadi.sk/i/lzCxOVbnkFfZc) in V. Lempitsky optimization course.
* [Overview of different LP solvers](https://medium.com/opex-analytics/optimization-modeling-in-python-pulp-gurobi-and-cplex-83a62129807a)
* [TED talks watching optimization](https://www.analyticsvidhya.com/blog/2017/10/linear-optimization-in-python/)
* [Overview of ellipsoid method](https://www.stat.cmu.edu/~ryantibs/convexopt-F13/scribes/lec15.pdf)
* [Comprehensive overview of linear programming](http://www.mit.edu/~kircher/lp.pdf)
* [Converting LP to a standard form](https://sites.math.washington.edu/~burke/crs/407/lectures/L4-lp_standard_form.pdf)
