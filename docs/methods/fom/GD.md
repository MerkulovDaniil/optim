---
layout: default
title: Gradient descent
parent: First order methods
grand_parent: Methods
nav_order: 1
bibtex: |
  @article{cauchy1847methode,
  title={M{\'e}thode g{\'e}n{\'e}rale pour la r{\'e}solution des systemes d’{\'e}quations simultan{\'e}es}, author={Cauchy, Augustin},
  journal={Comp. Rend. Sci. Paris},
  volume={25},
  number={1847},
  pages={536--538},
  year={1847}
  }
file: /assets/files/GD.pdf
---

# Summary

A classical problem of function minimization is considered. 

$$
\tag{GD}
x_{k+1} = x_k - \eta_k\nabla f(x_k)
$$

* The bottleneck (for almost all gradient methods) is choosing step-size, which can lead to the dramatic difference in method's behavior. 
* One of the theoretical suggestions: choosing stepsize inversly proportional to the gradient Lipschitz constant $$\eta_k = \dfrac{1}{L}$$
* In huge-scale applications the cost of iteration is usually defined by the cost of gradient calculation (at least $$\mathcal{O}(p)$$)
* If function has Lipschitz-continious gradient, then method could be rewritten as follows:

$$ \begin{align*}x_{k+1} &= x_{k}-\dfrac{1}{L} \nabla f\left(x_{k}\right)= \\
&= \arg \min\limits_{x \in \mathbb{R}^{n}}\left\{f\left(x_{k}\right)+\left\langle\nabla f\left(x_{k}\right), x-x_{k}\right\rangle+\frac{L}{2}\left\|x-x_{k}\right\|_{2}^{2}\right\} \end{align*}$$

# Intuition
## Direction of local steepest descent
Let's consider a linear approximation of the differentiable function $$f$$ along some direction $$h, \|h\|_2 = 1$$:

$$
f(x + \eta h) = f(x) + \eta \langle f'(x), h \rangle + o(\eta)
$$

We want $$h$$ to be a decreasing direction:

$$
f(x + \eta h) < f(x)
$$

$$
f(x) + \eta \langle f'(x), h \rangle + o(\eta) < f(x)
$$

and going to the limit at $$\eta \rightarrow 0$$:

$$
\langle f'(x), h \rangle \leq 0
$$

Also from Cauchy–Bunyakovsky–Schwarz inequality:

$$
|\langle f'(x), h \rangle | \leq \| f'(x) \|_2 \| h \|_2 \;\;\;\to\;\;\; \langle f'(x), h \rangle \geq -\| f'(x) \|_2 \| h \|_2 = -\| f'(x) \|_2
$$

Thus, the direction of the antigradient

$$
h = -\dfrac{f'(x)}{\|f'(x)\|_2}
$$

gives the direction of the **steepest local** decreasing of the function $$f$$.

The result of this method is

$$
x_{k+1} = x_k - \eta f'(x_k)
$$

## Gradient flow ODE

Let's consider the following ODE, which is referred as Gradient Flow equation.

$$
\label{GF}
\frac{dx}{dt} = -f'(x(t))
$$

and discretize it on a uniform grid with $$\eta$$ step:

$$
\frac{x_{k+1} - x_k}{\eta} = -f'(x_k),
$$

where $$x_k \equiv x(t_k)$$ and $$\eta = t_{k+1} - t_k$$ - is the grid step.

From here we get the expression for $$x_{k+1}$$

$$
x_{k+1} = x_k - \eta f'(x_k),
$$

which is exactly gradient descent.

## Necessary local minimum condition

$$
\begin{align*}
& f'(x) = 0\\
& -\eta f'(x) = 0\\
& x - \eta f'(x) = x\\
& x_k - \eta f'(x_k) = x_{k+1}
\end{align*}
$$

This is, surely, not a proof at all, but some kind of intuitive explanation.

## Minimizer of Lipschitz parabola

Some general highlights about Lipschitz properties are needed for explanation. If a function $$f: \mathbb{R}^n \to \mathbb{R}$$ is  continuously differentiable and its gradient satisfies Lipschitz conditions with constant $$L$$, then $$\forall x,y \in \mathbb{R}^n$$:

$$
|f(y) - f(x) - \langle \nabla f(x), y-x \rangle| \leq \frac{L}{2} \|y-x\|^2,
$$

which geometrically means, that if we'll fix some point $$x_0 \in \mathbb{R}^n$$ and define two parabolas:

$$
\phi_1(x) = f(x_0) + \langle \nabla f(x_0), x - x_0 \rangle - \frac{L}{2} \|x-x_0\|^2,
$$

$$
\phi_2(x) = f(x_0) + \langle \nabla f(x_0), x - x_0 \rangle + \frac{L}{2} \|x-x_0\|^2.
$$

Then 

$$
\phi_1(x) \leq f(x) \leq \phi_2(x) \forall x \in \mathbb{R}^n.
$$

Now, if we have global upper bound on the function, in a form of parabola, we can try to go directly to its minimum.

$$
\begin{align*}
& \phi_2(x) = 0 \\
& \nabla f(x_0) + L (x^* - x_0) = 0 \\
& x^* = x_0 - \frac{1}{L}\nabla f(x_0) \\
& x_{k+1} = x_k - \frac{1}{L} \nabla f(x_k)
\end{align*}
$$

![](../lipschitz_parabola.svg)

This way leads to the $$\frac{1}{L}$$ stepsize choosing. However, often the $$L$$ constant is not known.

But if the function is twice continuously differentiable and its gradient has Lipchitz constant $$L$$, we can derive a way to estimate this constant $$\forall x \in \mathbb{R}^n$$:

$$
\|\nabla^2 f(x) \| \leq L
$$

or

$$
-L I_n \preceq \nabla^2 f(x) \preceq L I_n
$$

# Stepsize choosing strategies

Stepsize choosing strategy $$\eta_k$$ significantly affects convergence. General {%include link.html title='Line search'%} algorithms might help in choosing scalar parameter. 
## Constant stepsize

For $$f \in C_L^{1,1}$$:

$$
\eta_k = \eta
$$

$$
f(x_k) - f(x_{k+1}) \geq \eta \left(1 - \frac{1}{2}L\eta \right) \|\nabla f(x_k)\|^2
$$

With choosing $$\eta = \frac{1}{L}$$, we have:

$$
f(x_k) - f(x_{k+1}) \geq \dfrac{1}{2L}\|\nabla f(x_k)\|^2
$$

## Fixed sequence

$$
\eta_k = \dfrac{1}{\sqrt{k+1}}
$$

The latter 2 strategies are the simplest in terms of implementation and analytical analysis. It is clear that this approach does not often work very well in practice (the function geometry is not known in advance).

## Exact line search aka steepest descent

$$
\eta_k = \text{arg}\min_{\eta \in \mathbb{R^+}} f(x_{k+1}) = \text{arg}\min_{\eta \in \mathbb{R^+}} f(x_k - \eta \nabla f(x_k))
$$

More theoretical than practical approach. It also allows you to analyze the convergence, but often exact line search can be difficult if the function calculation takes too long or costs a lot.

Interesting theoretical property of this method is that each following iteration is orthogonal to the previous one:

$$
\eta_k = \text{arg}\min_{\eta \in \mathbb{R^+}} f(x_k - \eta \nabla f(x_k))
$$

Optimality conditions:

$$
\nabla f(x_{k+1})^\top \nabla f(x_k) = 0
$$

## Goldstein-Armijo

This strategy of inexact line search works well in practice, as well as it has the following geometric interpretation:

Let's consider the following scalar function while being at a specific point of $$x_k$$: 

$$
\phi(\eta) = f(x_k - \eta\nabla f(x_k)), \eta \geq 0
$$

consider first order approximation of  $$\phi(\eta)$$:

$$
\phi(\eta) \approx f(x_k) - \eta\nabla f(x_k)^\top \nabla f(x_k)
$$

Let's consider also 2 linear scalar functions $$\phi_1(\eta), \phi_2(\eta)$$:

$$
\phi_1(\eta) = f(x_k) - \alpha \eta \|\nabla f(x_k)\|^2
$$
and
$$
\phi_2(\eta) = f(x_k) - \beta \eta \|\nabla f(x_k)\|^2
$$

Note, that Goldstein-Armijo conditions determine the location of the function $$\phi(\eta)$$ between $$\phi_1(\eta)$$ and $$\phi_2(\eta)$$. Typically, we choose $$\alpha = \rho$$ and $$\beta = 1 - \rho$$,while $$ \rho \in (0.5, 1)$$

![](../backtracking.svg)

# Convergence analysis
## Quadratic case

# Bounds

| Conditions | $\Vert f(x_k) - f(x^*)\Vert \leq$ | Type of convergence | $\Vert x_k - x^* \Vert \leq$ |
| ---------- | ---------------------- | ------------------- | --------------------- |
| Convex<br/>Lipschitz-continuous function($G$) | $\mathcal{O}\left(\dfrac{1}{k} \right) \; \dfrac{GR}{k}$ | Sublinear |                       |
| Convex<br/>Lipschitz-continuous gradient ($L$) | $\mathcal{O}\left(\dfrac{1}{k} \right) \; \dfrac{LR^2}{k}$ | Sublinear |                       |
| $\mu$-Strongly convex<br/>Lipschitz-continuous gradient($L$) |                        | Linear | $(1 - \eta \mu)^k R^2$ |
| $\mu$-Strongly convex<br/>Lipschitz-continuous hessian($M$) |                        | Locally linear<br /> $R < \overline{R}$ | $\dfrac{\overline{R}R}{\overline{R} - R} \left( 1 - \dfrac{2\mu}{L+3\mu}\right)$ |

* $$R = \| x_0 - x^*\| $$ - initial distance
* \$$\overline{R} = \dfrac{2\mu}{M}$$

# Materials

* [The zen of gradient descent. Moritz Hardt](http://blog.mrtz.org/2013/09/07/the-zen-of-gradient-descent.html)
* [Great visualization](http://fa.bianp.net/teaching/2018/eecs227at/gradient_descent.html)
