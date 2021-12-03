---
layout: default
parent: Theory
title: Duality
nav_order: 10
---

# Duality

Duality lets us associate to any constrained optimization problem a concave maximization problem, whose solutions lower bound the optimal value of the original problem. What is interesting is that there are cases, when one can solve the primal problem by first solving the dual one. Now, consider a general constrained optimization problem:

$$
\text{ Primal: }f(x) \to \min\limits_{x \in S}  \qquad \text{ Dual: } g(y) \to \max\limits_{y \in \Omega} 
$$

We'll build $g(y)$, that preserves the uniform bound:

$$
g(y) \leq f(x) \qquad \forall x \in S, \forall y \in \Omega
$$

As a consequence:

$$
\max\limits_{y \in \Omega} g(y) \leq \min\limits_{x \in S} f(x)  
$$

We'll consider one (of the many) possible way to construct $g(y)$ in case, when we have a general mathematical programming problem with functional constraints:

$$
\begin{split}
& f_0(x) \to \min\limits_{x \in \mathbb{R}^n}\\
\text{s.t. } & f_i(x) \leq 0, \; i = 1,\ldots,m\\
& h_i(x) = 0, \; i = 1,\ldots, p
\end{split}
$$

And the Lagrangian, associated with this problem:

$$
L(x, \lambda, \nu) = f_0(x) + \sum\limits_{i=1}^m \lambda_i f_i(x) + \sum\limits_{i=1}^p\nu_i h_i(x) = f_0(x) + \lambda^\top f(x) + \nu^\top h(x)
$$


We define the Lagrange dual function (or just dual function) $$g: \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$$ as the minimum value of the Lagrangian over $$x$$: for $$\lambda \in \mathbb{R}^m, \nu \in \mathbb{R}^p$$

$$
g(\lambda, \nu) = \inf_{x\in \textbf{dom} f_0} L(x, \lambda, \nu) = \inf_{x\in \textbf{dom} f_0} \left( f_0(x) +\sum\limits_{i=1}^m \lambda_i f_i(x) + \sum\limits_{i=1}^p\nu_i h_i(x) \right)
$$

When the Lagrangian is unbounded below in $$x$$, the dual function takes on the value $$−\infty$$. Since the dual function is the pointwise infimum of a family of affine functions of $$(\lambda, \nu)$$, it is concave, even when the original problem is not convex.


The dual function yields lower bounds on the optimal value $$p^*$$ of the original problem. For any $$\lambda \succeq 0, \nu$$:

$$
g(\lambda, \nu) \leq p^*
$$

Suppose some $$\hat{x}$$ is a feasible point ($$\hat{x} \in S$$) for the original problem, i.e., $$f_i(\hat{x}) \leq 0$$ and $$h_i(\hat{x}) = 0, \; λ \succeq 0$$. Then we have:

$$
L(\hat{x}, \lambda, \nu) = f_0(\hat{x}) + \lambda^\top f(\hat{x}) + \nu^\top h(\hat{x}) \leq f_0(\hat{x})
$$

Hence

$$
g(\lambda, \nu) = \inf_{x\in \textbf{dom} f_0} L(x, \lambda, \nu) \leq L(\hat{x}, \lambda, \nu)  \leq f_0(\hat{x})
$$

A natural question is: what is the *best* lower bound that can be obtained from the Lagrange dual function? 
This leads to the following optimization problem:

$$
\begin{split}
& g(\lambda, \nu) \to \max\limits_{\lambda \in \mathbb{R}^m, \; \nu \in \mathbb{R}^p }\\
\text{s.t. } & \lambda \succeq 0
\end{split}
$$


The term "dual feasible", to describe a pair $$(\lambda, \nu)$$ with $$\lambda \succeq 0$$ and $$g(\lambda, \nu) > −\infty$$, now makes sense. It means, as the name implies, that $$(\lambda, \nu)$$ is feasible for the dual problem. We refer to $$(\lambda^*, \nu^*)$$ as dual optimal or optimal Lagrange multipliers if they are optimal for the above problem.


## Summary

|  | Primal | Dual |
|:-----------:|:---------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------:|
| Function | $$f_0(x)$$ | $$g(\lambda, \nu) = \min\limits_{x \in \textbf{dom} f_0} L(x, \lambda, \nu)$$ |
| Variables | $$x \in \textbf{dom} f_0 \subseteq \mathbb{R^n}$$ | $$\lambda \in \mathbb{R}^m_{+}, \nu \in \mathbb{R}^m$$ |
| Constraints | $$\begin{split} & f_i(x) \leq 0, \; i = 1,\ldots,m\\ & h_i(x) = 0, \; i = 1,\ldots, p \end{split}$$ | $$\lambda_i \geq 0, \forall i \in \overline{1,m}$$ |
| Problem | $$\begin{split}& f_0(x) \to \min\limits_{x \in \mathbb{R}^n}\\ \text{s.t. } & f_i(x) \leq 0, \; i = 1,\ldots,m\\ & h_i(x) = 0, \; i = 1,\ldots, p \end{split}$$ | $$\begin{split}  g(\lambda, \nu) &\to \max\limits_{\lambda \in \mathbb{R}^m, \nu \in \mathbb{R}^p }\\ \text{s.t. } & \lambda \succeq 0 \end{split}$$ | 
| Optimal | $$\begin{split} &x^* \text{ if feasible},  \\ &p^* = f_0(x^*)\end{split}$$ | $$\begin{split} &\lambda^*, \nu^* \text{ if } \max \text{ is achieved},  \\ &d^* = g(\lambda^*, \nu^*)\end{split}$$ |


## Weak duality
It is common to name this relation between optimals of primal and dual problems as weak duality. For problem, we have: 

$$
p^* \geq d^*
$$

While the difference between them is often called **duality gap:** 

$$
p^* - d^* \geq 0
$$

Note, that we always have weak duality, if we've formulated primal and dual problem. It means, that if we have managed to solve the dual problem (which is always convex, no matter whether the initial problem was or not), then we have some lower bound. Surprisingly, there are some notable cases, when these solutions are equal.

## Strong duality
Strong duality happens if duality gap is zero: 

$$
p^∗ = d^*
$$

Notice: both $$p^*$$ and $$d^*$$ may be $$+ \infty$$. 
* Several sufficient conditions known!
* “Easy” necessary and sufficient conditions: unknown.

## Useful features
* **Construction of lower bound on solution of the direct problem.**

	It could be very complicated to solve the initial problem. But if we have the dual problem, we can take an arbitrary $y \in \Omega$ and substitute it in $g(y)$ - we'll immediately obtain some lower bound.
* **Checking for the problem's solvability and attainability of the solution.** 

	From the inequality $\max\limits_{y \in \Omega} g(y) \leq \min\limits_{x \in S} f_0(x) $ follows: if $\min\limits_{x \in S} f_0(x) = -\infty$, then $\Omega = \varnothing$ and vice versa.
* **Sometimes it is easier to solve a dual problem than a primal one.** 

	In this case, if the strong duality holds: $g(y^∗) = f_0(x^∗)$ we lose nothing.
* **Obtaining a lower bound on the function's residual.** 

	$f_0(x) - f^∗ \leq f_0(x) - g(y)$ for an arbitrary $y \in \Omega$ (suboptimality certificate). Moreover, $$p^* \in [g(y), f_0(x)], d^* \in [g(y), f_0(x)]$$
* **Dual function is always concave**

	As a pointwise minimum of affine functions.

# Examples

## Simple projection onto simplex with duality

To find the Euclidean projection of $$x \in \mathbb{R}^n$$ onto probability simplex $$\mathcal{P} = \{z \in \mathbb{R}^n \mid z \succeq 0, \mathbf{1}^\top z = 1\}$$, we solve the following problem:

$$
\begin{split}
& \dfrac{1}{2}\|y - x\|_2^2 \to \min\limits_{y \in \mathbb{R}^{n} \succeq 0}\\
\text{s.t. } & \mathbf{1}^\top y = 1
\end{split}
$$

*Hint:* Consider the problem of minimizing $$\dfrac{1}{2}\|y - x\|_2^2 $$ subject to subject to $$y \succeq 0, \mathbf{1}^\top y = 1$$. Form the partial Lagrangian

$$
L(y, \nu) = \dfrac{1}{2}\|y - x\|_2^2 +\nu(\mathbf{1}^\top y - 1),
$$

leaving the constraint $$y \succeq 0$$ implicit. Show that $$y = (x − \nu \mathbf{1})_+$$ minimizes $$L(y, \nu)$$ over $$y \succeq 0$$.

## Underdetermined Linear least squares

### Problem

$$
\begin{split}
& x^\top x \to \min\limits_{x \in \mathbb{R}^{n} }\\
\text{s.t. } & Ax = b
\end{split}
$$

### Lagrangian

$$
L(x,\nu) = x^\top x + \nu^\top (Ax - b)
$$


### Dual function

$$
g(\nu) = \min\limits_{x \in \mathbb{R}^n} L(x, \nu) 
$$

Setting gradient to zero to find minimum 🤔:

$$
\nabla_x L (x, \nu) = 2x + \nu^\top A = 2x + A^\top \nu = 0 \; \to \; x = -\frac{1}{2}A^\top \nu
$$

$$
g(\nu) = \frac{1}{4} \nu^\top A A^\top \nu + \nu^\top (-\frac{1}{2} A A^\top \nu - b) = -\frac{1}{4} \nu^\top A A^\top \nu - b^\top \nu
$$

Here we see lower bound property:

$$
p^* \geq -\frac{1}{4} \nu^\top A A^\top \nu - b^\top \nu, \quad \forall \nu
$$

Let's solve the dual problem:

$$
d^* = b^\top \left( A A^\top \right)^{-1}b
$$

Calculate the primal optimal and check whether this problem has strong duality or not.

## LP duality. Standard form

$$
\begin{split}
& c^\top x \to \min\limits_{x \in \mathbb{R}^{n} }\\
\text{s.t. } & Ax = b \\
& x \succeq 0 \\
\end{split}
$$

## LP duality. Inequality form

$$
\begin{split}
& c^\top x \to \min\limits_{x \in \mathbb{R}^{n} }\\
\text{s.t. } & Ax \preceq b 
\end{split}
$$

## A nonconvex quadratic problem with strong duality

On rare occasions strong duality obtains for a nonconvex problem. As an important example, we consider the problem of minimizing a nonconvex quadratic function over the unit ball

$$
\begin{split}
& x^\top A x  + 2b^\top x\to \min\limits_{x \in \mathbb{R}^{n} }\\
\text{s.t. } & x^\top x \leq 1 
\end{split}
$$


where $$A \in \mathbb{S}^n, A \nsucceq 0$$ and $$b \in \mathbb{R}^n$$. Since $$A \nsucceq 0$$, this is not a convex problem. This problem is sometimes called the trust region problem, and arises in minimizing a second-order approximation of a function over the unit ball, which is the region in which the approximation is assumed to be approximately valid.

### Lagrangian and dual function

$$
L(x, \lambda) = x^\top A x + 2 b^\top x + \lambda (x^\top x - 1) = x^\top( A + \lambda I)x + 2 b^\top x - \lambda
$$

$$
g(\lambda) = \begin{cases} -b^\top(A + \lambda I)^{\dagger}b - \lambda &\text{ if } A + \lambda I \succeq 0 \\ -\infty, &\text{ otherwise}  \end{cases}
$$

### Dual problem

$$
\begin{split}
& -b^\top(A + \lambda I)^{\dagger}b - \lambda \to \max\limits_{\lambda \in \mathbb{R}}\\
\text{s.t. } & A + \lambda I \succeq 0
\end{split}
$$

$$
\begin{split}
& -\sum\limits_{i=1}^n \dfrac{(q_i^\top b)^2}{\lambda_i + \lambda} - \lambda  \to \max\limits_{\lambda \in \mathbb{R}}\\
\text{s.t. } & \lambda \geq - \lambda_{min}(A)
\end{split}
$$



## Connection of Fenchel and Lagrange duality

$$
\begin{split}
& f_0(x) = \sum_{i=1}^n f_i(x_i)\to \min\limits_{x \in \mathbb{R}^{n} }\\
\text{s.t. } & a^\top x = b 
\end{split}
$$

The dual problem is thus 

$$
\begin{split}
& -b \nu - \sum_{i=1}^n f_i^*(-\nu a_i)  \to \max\limits_{\nu \in \mathbb{R}}\\
\text{s.t. } & \lambda \geq - \lambda_{min}(A)
\end{split}
$$

with (scalar) variable $$\nu \in \mathbb{R}$$. Now suppose we have found an optimal dual variable $$\nu^*$$ (There are several simple methods for solving a convex problem with one scalar variable, such as the bisection method.). It is very easy to recover the optimal value for the primal problem.

## Fenchel - Rockafellar problem
### Problem
Let $$f: E \to \mathbb{R}$$ and $$g: G \to \mathbb{R}$$ — function, defined on the sets $$E$$ and $$G$$ in Euclidian Spaces $$V$$ and $$W$$ respectively. Let $$f^*:E_* \to \mathbb{R}, g^*:G_* \to \mathbb{R}$$ be the conjugate functions to the $$f$$ and $$g$$ respectively. Let $$A: V \to W$$ — linear mapping. We call Fenchel - Rockafellar problem the following minimization task: 

$$
f(x) + g(Ax) \to \min\limits_{x \in E \cap A^{-1}(G)}
$$

where $$A^{-1}(G) := \{x \in V : Ax \in G\}$$ — preimage of $$G$$.
We'll build the dual problem using variable separation. Let's introduce new variable $$y = Ax$$. The problem could be rewritten:

$$
\begin{split}
& f(x) + g(y) \to \min\limits_{x \in E, \; y \in G }\\
\text{s.t. } & Ax = y
\end{split}
$$

### Lagrangian

$$
L(x,y, \lambda) =  f(x) + g(y) + \lambda^\top (Ax - y)
$$

### Dual function

$$
\begin{split}
g_d(\lambda) &= \min\limits_{x \in E, \; y \in G} L(x,y, \lambda) \\
&= \min\limits_{x \in E}\left[ f(x) + (A^*\lambda)^\top x \right] + \min\limits_{y \in G} \left[g(y) - \lambda^\top y\right] = \\
&= -\max\limits_{x \in E}\left[(-A^*\lambda)^\top x - f(x) \right] - \max\limits_{y \in G} \left[\lambda^\top y - g(y)\right]
\end{split}
$$

Now, we need to remember the definition of the {% include link.html title='Conjugate function'%}:

$$
\sup_{y \in G}\left[\lambda^\top y - g(y)\right] = \begin{cases} g^*(\lambda), &\text{ if } \lambda \in G_*\\ +\infty, &\text{ otherwise} \end{cases}
$$

$$
\sup_{x \in E}\left[(-A^*\lambda)^\top x - f(x) \right] = \begin{cases} f^*(-A^*\lambda), &\text{ if } \lambda \in (-A^*)^{-1}(E_*)\\ +\infty, &\text{ otherwise} \end{cases}
$$

So, we have:

$$
\begin{split}
g_d(\lambda) &= \min\limits_{x \in E, y \in G} L(x,y, \lambda) = \\
&= \begin{cases} -g^*(\lambda) - f^*(-A^*\lambda) &\text{ if } \lambda \in G_* \cap (-A^*)^{-1}(E_*)\\ -\infty, &\text{ otherwise}  \end{cases}
\end{split}
$$

which allows us to formulate one of the most important theorems, that connects dual problems and conjugate functions:

**Fenchel - Rockafellar theorem** Let $$f: E \to \mathbb{R}$$ and $$g: G \to \mathbb{R}$$ — function, defined on the sets $$E$$ and $$G$$ in Euclidian Spaces $$V$$ and $$W$$ respectively. Let $$f^*:E_* \to \mathbb{R}, g^*:G_* \to \mathbb{R}$$ be the conjugate functions to the $$f$$ and $$g$$ respectively. Let $$A: V \to W$$ — linear mapping. Let $$p^*, d^* \in [- \infty, + \infty]$$ - optimal values of primal and dual problems:

$$
p^* = f(x) + g(Ax) \to \min\limits_{x \in E \cap A^{-1}(G)}
$$

$$
d^* = f^*(-A^*\lambda) + g^*(\lambda) \to \min\limits_{\lambda \in G_* \cap (-A^*)^{-1}(E_*)},
$$

Then we have weak duality: $$p^* \geq d^*$$. Furthermore, if the functions $$f$$ and $$g$$ are convex and $$A(\mathbf{relint}(E)) \cap \mathbf{relint}(G) \neq \varnothing $$, then we have strong duality: $$p^* = d^*$$. While points $$x^* \in E \cap A^{-1}(G)$$ and $$\lambda^* \in G_* \cap (-A^*)^{-1}(E_*)$$ are optimal values for primal and dual problem if and only if:

$$
\begin{split}
-A^*\lambda^* &\in \partial f(x^*) \\
\lambda^* &\in \partial g(Ax^*)
\end{split}
$$

Convex case is especially important since if we have Fenchel - Rockafellar problem with parameters $$(f, g, A)$$, than the dual problem has the form $$(f^*, g^*, -A^*)$$.





# References

* [Convex Optimization — Boyd & Vandenberghe @ Stanford](http://web.stanford.edu/class/ee364a/lectures/duality.pdf)
* [Course Notes for EE227C. Lecture 13](https://ee227c.github.io/notes/ee227c-lecture13.pdf)
* [Course Notes for EE227C. Lecture 14](https://ee227c.github.io/notes/ee227c-lecture14.pdf)
* {% include link.html title='Optimality conditions. KKT' %}
* [Seminar 7 @ CMC MSU](http://www.machinelearning.ru/wiki/images/7/7f/MOMO18_Seminar7.pdf)
* [Seminar 8 @ CMC MSU](http://www.machinelearning.ru/wiki/images/1/15/MOMO18_Seminar8.pdf)
* [Convex Optimization @ Berkley - 10th lecture](http://suvrit.de/teach/ee227a/lect10.pdf)
