---
layout: default
parent: Theory
title: Duality
nav_order: 10
---

# Duality

Duality lets us associate to any constrained optimization problem, a concave maximization problem whose solutions lower bound the optimal value of the original problem. What is interesting is there are cases, when one can solve the primal problem by first solving the dual one. Now, consider general constrained optimization problem:

$$
\text{ Primal: }f(x) \to \min\limits_{x \in S}  \qquad \text{ Dual: } g(y) \to \max\limits_{y \in \Omega} 
$$

We'll build $g(y)$, that preserves uniform bound:

$$
g(y) \leq f(x) \qquad \forall x \in S, \forall y \in \Omega
$$

As a consequence:

$$
\max\limits_{y \in \Omega} g(y) \leq \min\limits_{x \in S} f(x)  
$$

We'll consider one (of the many) possible way to construct $g(y)$ in case, when we have general mathematical programming problem with functional constraints:

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n}\\
\text{s.t. } & g_i(x) \leq 0, \; i = 1,\ldots,m\\
& h_j(x) = 0, \; j = 1,\ldots, p
\end{split}
$$

And the Lagrangian, associated with this problem:

$$
L(x, \lambda, \mu) = f(x) + \sum\limits_{j=1}^p\lambda_j h_j(x) + \sum\limits_{i=1}^m \mu_i g_i(x) = f(x) + \lambda^\top h(x) + \mu^\top g(x)
$$

Now, we notice, that the original problem is equivalent to the following *unconstraint* problem:

$$
\min\limits_{x \in S} f(x)  = \min\limits_{x \in \mathbb{R}^n} \max\limits_{\lambda \in \mathbb{R}^p, \mu \in \mathbb{R}^m_{+}} L(x, \lambda, \mu)
$$

Because $\lambda$ don't affect the equality entries (since they all are equal to zero), while $\mu$ is positive and decreases Lagrangian (since all inequality entries are non - positive). Moreover, it can be seen, that:

$$
\max\limits_{\lambda \in \mathbb{R}^p, \mu \in \mathbb{R}^m_{+}} \min\limits_{x \in \mathbb{R}^n} L(x, \lambda, \mu) \leq \min\limits_{x \in \mathbb{R}^n} \max\limits_{\lambda \in \mathbb{R}^p, \mu \in \mathbb{R}^m_{+}} L(x, \lambda, \mu)
$$

Now, we denote the inner optimization as follows:

$$
g(\lambda, \mu) = \min\limits_{x \in \mathbb{R}^n} L(x, \lambda, \mu)
$$

## Summary

|  | Primal | Dual |
|:-----------:|:---------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------:|
| Function | $$f(x)$$ | $$g(\lambda, \mu) = \min\limits_{x \in \mathbb{R}^n} L(x, \lambda, \mu)$$ |
| Variables | $$x \in \mathbb{R^n}$$ | $$\lambda \in \mathbb{R}^p, \mu \in \mathbb{R}^m_{+}$$ |
| Constraints | $$\begin{split} & g_i(x) \leq 0, \; i = 1,\ldots,m\\ & h_j(x) = 0, \; j = 1,\ldots, p \end{split}$$ | $$\mu_i \geq 0, \forall i \in \overline{1,m}$$ |
| Problem | $$\min\limits_{x \in S}f(x)$$ | $$\max\limits_{\lambda \in \mathbb{R}^p, \mu \in \mathbb{R}^m_{+}}\min\limits_{x \in \mathbb{R}^n} L(x, \lambda, \mu)$$ |
| Optimal | $$\begin{split} &x^* \text{ if feasible},  \\ &p^* = f(x^*)\end{split}$$ | $$\begin{split} &\lambda^*, \mu^* \text{ if } \max \text{ is achieved},  \\ &d^* = g(\lambda^*, \mu^*)\end{split}$$ |

## Useful features
* **Construction of lower bound on solution of the direct problem.**

	It could be very complicated to solve the initial problem. But if we have the dual problem, we can take an arbitrary $y \in \Omega$ and substitute it in $g(y)$ - we'll immediately obtain some lower bound.
* **Checking for the problem's solvability and attainability of the solution.** 

	From the inequality $\max\limits_{y \in \Omega} g(y) \leq \min\limits_{x \in S} f(x) $ follows: if $\min\limits_{x \in S} f(x) = -\infty$, then $\Omega = \varnothing$ and vice versa.
* **Sometimes it is easier to solve a dual problem than a primal one.** 

	In this case, if the strong duality holds: $g(y^∗) = f(x^∗)$ we lose nothing.
* **Obtaining a lower bound on the function's residual.** 

	$f(x) - f^∗ \leq f(x) - g(y)$ for an arbitrary $y \in \Omega$
* **Dual function is always concave**

	As a pointwise minimum of affine functions.

# Weak duality
It is common to name this relation between optimals of primal and dual problems as weak duality. For problem, we have: 

$$
p^* \geq d^*
$$

While the difference between them is often called **duality gap:** 

$$
p^* - d^* \geq 0
$$

Note, that we always have weak duality, if we've formulated primal and dual problem. It means, that if we were managed to solve dual problem (which is always convex, no matter the initial problem was or not), than we have some lower bound. Surprisingly, there are some notable cases, when these solutions are equal.

# Strong duality
Strong duality if duality gap is zero: 

$$
p^∗ = d^*
$$

Notice: both $$p^*$$ and $$d^*$$ may be $$+ \infty$$. 
* Several sufficient conditions known!
* “Easy” necessary and sufficient conditions: unknown.

## Slater's sufficient condition
**Theorem** Let the primal problem be the {% include link.html title='Convex optimization problem' %}. If there is a feasible point such that is strictly feasible for the non-affine constraints (and merely feasible for affine, linear ones), then strong duality holds. Moreover, in this case, the dual optimal is attained (i.e., $$d^* > -\infty$$). 

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n}\\
\text{s.t. } & g_i(x) \leq 0, \; i = 1,\ldots,m\\
& Ax = b
\end{split}
$$

Slater's condition: There exists $$x \in \mathbf{relint} S$$ s.t. $$g_i(x) < 0 , Ax = b, i = 1,\ldots,m$$, i.e. there is a **strictly feasible point**.

**Counterexample**

$$
\begin{split}
& e^{-x} \to \min\limits_{(x,y) \in \mathbb{R}^{2} }\\
\text{s.t. } & \frac{x^2}{y} \leq 0\\
& y > 0
\end{split}
$$

Only feasible line is $$x = 0$$, therefore we know optimal value of the primal problem $$p^* = e^{0} = 1$$. Let's construct the Lagrangian:

$$
L(x,y, \mu) = e^{-x} + \mu_1 \frac{x^2}{y} + \mu_2 (-y)
$$

Then, dual function is:

$$
g(\mu) = \min\limits_{(x,y) \in \mathbb{R}^2} L(x, y, \mu) = e^{-x} + \mu_1 \frac{x^2}{y} + \mu_2 (-y) = \begin{cases} 0, &\mu_1 \geq 0\\ - \infty, &\mu_1 < 0\end{cases}
$$

Dual problem is formulating the following way:

$$
d^* = \max_{\mu \in \mathbb{R}^2_+} 0  = 0
$$

Here, we clearly see unit dual gap: $$p^* - d^* = 1$$ and lack of strictly feasible solutions.
# Examples
## Least squares
### Problem

$$
\begin{split}
& x^\top x \to \min\limits_{x \in \mathbb{R}^{n} }\\
\text{s.t. } & Ax = b
\end{split}
$$

### Lagrangian

$$
L(x,\lambda) = x^\top x + \lambda^\top (Ax - b)
$$


### Dual function

$$
g(\lambda) = \min\limits_{x \in \mathbb{R}^n} L(x, \lambda) 
$$

Setting gradient to zero to find minimum 🤔:

$$
\nabla_x L (x, \lambda) = 2x + \lambda^\top A = 2x + A^\top \lambda = 0 \; \to \; x = -\frac{1}{2}A^\top \lambda
$$

$$
g(\lambda) = \frac{1}{4} \lambda^\top A A^\top \lambda + \lambda^\top (-\frac{1}{2} A A^\top \lambda - b) = -\frac{1}{4} \lambda^\top A A^\top \lambda - b^\top \lambda
$$

Here we see lower bound property:

$$
p^* \geq -\frac{1}{4} \lambda^\top A A^\top \lambda - b^\top \lambda, \quad \forall \lambda
$$

Let's solve the dual problem:

$$
d^* = b^\top \left( A A^\top \right)^{-1}b
$$

Calculate the primal optimal and check whether this problem has strong duality or not.

## Fenchel - Rockafellar problem
### Problem
Let $$f: E \to \mathbb{R}$$ and $$g: G \to \mathbb{R}$$ — function, defined on the sets $$E$$ and $$G$$ in Euclidian Spaces $$V$$ and $$W$$ respectively. Let $$f^*:E_* \to \mathbb{R}, g^*:G_* \to \mathbb{R}$$ be the conjugate functions to the $$f$$ and $$g$$ respectively. Let $$A: V \to W$$ — linear mapping. We call Fenchel - Rockafellar problem the following minimization task: 

$$
f(x) + g(Ax) \to \min\limits_{x \in E \cap A^{-1}(G)}
$$

where $$A^{-1}(G) := \{x \in V : Ax \in G\}$$ — preimage of $$G$$.
We'll build the dual problem using variable separation. Let's introduce new variable $$y = Ax$$. The the problem could be rewritten:

$$
\begin{split}
& f(x) + g(y) \to \min\limits_{x \in E y \in G }\\
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
g_d(\lambda) &= \min\limits_{x \in E, y \in G} L(x,y, \lambda) \\
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

So, we have

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

Then we have weak duality: $$p^* \geq d^*$$. Furthermore, if the functions $$f$$ and $$g$$ are convex and $$A(\mathbf{relint}(E)) \cap \mathbf{relint}(G) \neq \varnothing $$, then we have strong duality: $$p^* = d^*$$. While points $$x^* \in E \cap A^{-1}(G)$$ and $$\lambda^* \in G_* \cap (-A^*)^{-1}(E_*)$$ are optimal values for primal and dual problem if and only if

$$
\begin{split}
-A^*\lambda^* &\in \partial f(x^*) \\
\lambda^* &\in \partial g(Ax^*)
\end{split}
$$

Convex case is especially important since if we have Fenchel - Rockafellar problem with parameters $$(f, g, A)$$, than the dual problem has the form $$(f^*, g^*, -A^*)$$

## Logistic regression
### Problem

$$
\begin{split}
& \sum\limits_{i=1}^m \ln\left(1 + e^{\langle a_i, x \rangle}\right) \to \min\limits_{x \in \mathbb{R}^n },
\end{split}
$$

where $$a_1, \ldots, a_m \in \mathbb{R}^n$$. Let $$f: \mathbb{R}^n \to \mathbb{R}$$ and $$g: \mathbb{R}^m \to \mathbb{R}$$ be the following functions

$$
f(x) = 0, \quad g(y) = \sum\limits_{i=1}^m \ln\left(1 + e^{y_i} \right)
$$

And let $$A: \mathbb{R}^n \to \mathbb{R}^m$$ be some linear map:

$$
Ax = \left(\langle a_i, x \rangle \right)_{1 \leq i \leq m}
$$

Then, we can introduce Fenchel - Rockafellar problem:

$$
\begin{split}
f^* &= \delta_{\{0\}} \\
g^*:[0,1]^m \to \mathbb{R}, g^*(\lambda) &= \sum\limits_{i=1}^m \text{Bin_Ent}\lambda_i\\
A^*: \mathbb{R}^m \to \mathbb{R}^n &= A^*\lambda = \sum\limits_{i=1}^m \lambda_i a_i \\
p &= \min_{x \in \mathbb{R}^n} f(x) + g(Ax)
\end{split}
$$

$$
\begin{split}
d = \min_{\lambda \in [0,1]^m} & \sum\limits_{i=1}^m \text{Bin_Ent}\lambda_i, \\
 & \sum\limits_{i=1}^m \lambda_i a_i = 0 
\end{split}
$$

## Entropy maximization
### Problem

$$
\begin{split}
& \sum_i x_i \ln x_i \to \min\limits_{x \in \mathbb{R}^{n} }\\
\text{s.t. } & Ax \leq b \\
& 1^\top x = 1 \\
& x > 0
\end{split}
$$

### Lagrangian

$$
L(x,y, \lambda, \mu) = \sum_i x_i \ln x_i + \lambda^\top (Ax - b)
$$

# References

* [Convex Optimization — Boyd & Vandenberghe @ Stanford](http://web.stanford.edu/class/ee364a/lectures/duality.pdf)
* [Course Notes for EE227C. Lecture 13](https://ee227c.github.io/notes/ee227c-lecture13.pdf)
* [Course Notes for EE227C. Lecture 14](https://ee227c.github.io/notes/ee227c-lecture14.pdf)
* {% include link.html title='Optimality conditions. KKT' %}
* [Seminar 7 @ CMC MSU](http://www.machinelearning.ru/wiki/images/7/7f/MOMO18_Seminar7.pdf)
* [Seminar 8 @ CMC MSU](http://www.machinelearning.ru/wiki/images/1/15/MOMO18_Seminar8.pdf)
* [Convex Optimization @ Berkley - 10th lecture](http://suvrit.de/teach/ee227a/lect10.pdf)
