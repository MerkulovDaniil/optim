---
layout: default
parent: Theory
title: Optimality conditions. KKT
nav_order: 8
---

# Background
## Extreme value (Weierstrass) theorem
Let $$S \subset \mathbb{R}^n$$ be compact set and $$f(x)$$ continuous function on $$S$$. 
So that, the point of the global minimum of the function $$f (x)$$ on $$S$$ exists.

![](../goodnews.png)

## Lagrange multipliers
Consider simple yet practical case of equality constraints:

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
\text{s.t. } & h_i(x) = 0, i = 1, \ldots, m
\end{split}
$$

The basic idea of Lagrange method implies switch from conditional to unconditional optimization through increasing the dimensionality of the problem:

$$
L(x, \lambda) = f(x) + \sum\limits_{i=1}^m \lambda_i h_i(x) \to \min\limits_{x \in \mathbb{R}^n, \lambda \in \mathbb{R}^m} \\
$$

# General formulations and conditions


$$
f(x) \to \min\limits_{x \in S}
$$

We say that the problem has a solution if the following set **is not empty**: $x^* \in S$, in which the minimum or the infimum of the given function is achieved.  

## Unconstrained optimization
### General case
Let $$f(x): \mathbb{R}^n \to \mathbb{R}$$ be a twice differentiable function.

$$
\tag{UP}
f(x) \to \min\limits_{x \in \mathbb{R}^n}
$$

If $$x^*$$ - is a local minimum of $$f(x)$$, then:

$$
\tag{UP:Necessary}
\nabla f(x^*) = 0
$$

If $$f(x)$$ at some point $$x^*$$ satisfies the following conditions:

$$
\tag{UP:Sufficient}
H_f(x^*) = \nabla^2 f(x^*) \succeq (\preceq) 0,
$$

then (if necessary condition is also satisfied) $$x^*$$ is a local minimum(maximum) of $$f(x)$$.

### Convex case
It should be mentioned, that in **convex** case (i.e., $$f(x)$$ is convex) necessary condition becomes sufficient. Moreover, we can generalize this result on the class of non-differentiable convex functions. 

Let $$f(x): \mathbb{R}^n \to \mathbb{R}$$ - convex function, then the point $$x^*$$ is the solution of $$\text{(UP)}$$ if and only if:

$$
0_n \in \partial f(x^*)
$$

One more important result for convex constrained case sounds as follows. If $$f(x): S \to \mathbb{R}$$ - convex function defined on the convex set $$S$$, then:
* Any local minima is the global one.
* The set of the local minimizers $$S^*$$ is convex.
* If $$f(x)$$ - strongly convex function, then $$S^*$$ contains only one single point $$S^* = x^*$$.

## Optimization with equality conditions
### Intuition
Things are pretty simple and intuitive in unconstrained problem. In this section we will add one equality constraint, i.e.

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
\text{s.t. } & h(x) = 0
\end{split}
$$

We will try to illustrate approach to solve this problem through the simple example with $$f(x) = x_1 + x_2$$ and $$h(x) = x_1^2 + x_2^2 - 2$$

![](../kkt_images/KKT_p009.svg)

![](../kkt_images/KKT_p010.svg)

![](../kkt_images/KKT_p011.svg)

![](../kkt_images/KKT_p012.svg)

![](../kkt_images/KKT_p013.svg)

![](../kkt_images/KKT_p014.svg)

![](../kkt_images/KKT_p015.svg)

![](../kkt_images/KKT_p016.svg)

![](../kkt_images/KKT_p017.svg)

Generally: in order to move from $$ x_F $$ along the budget set towards decreasing the function, we need to guarantee two conditions:

$$
\langle \delta x, \nabla h(x_F) \rangle = 0
$$

$$
\langle \delta x, - \nabla f(x_F) \rangle > 0
$$

Let's assume, that in the process of such a movement we have come to the point where
$$
\nabla f(x) = \lambda \nabla h(x)
$$

$$
\langle  \delta x, - \nabla f(x)\rangle = -\langle  \delta x, \lambda\nabla h(x)\rangle = 0  
$$

Then we came to the point of the budget set, moving from which it will not be possible to reduce our function. This is the local minimum in the limited problem :)
![](../kkt_images/KKT_p021.svg)

So let's define a Lagrange function (just for our convenience):

$$
L(x, \lambda) = f(x) + \lambda h(x)
$$

Then the point $$ x^* $$ be the local minimum of the problem described above, if and only if:

$$
\begin{split}
& \nabla_x L(x^*, \lambda^*) = 0 \text{ that's written above}\\
& \nabla_\lambda L(x^*, \lambda^*) = 0 \text{ condition of being in budget set}\\
& \langle y , \nabla^2_{xx} L(x^*, \lambda^*) y \rangle \geq 0, \;\;\; \forall y \in \mathbb{R}^n : \nabla h(x^*)^\top y = 0
\end{split}
$$

We should notice that $$L(x^*, \lambda^*) = f(x^*)$$.

### General formulation

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
\text{s.t. } & h_i(x) = 0, \; i = 1,\ldots, m 
\end{split}
$$

Solution 

$$
L(x, \lambda) = f(x) + \sum\limits_{i=1}^m\lambda_i g_i(x) = f(x) + \lambda^\top g(x)
$$

Let $$ f(x) $$ and $$ h_i(x) $$ be twice differentiable at the point $$ x^* $$ and continuously differentiable in some neighborhood $$ x^* $$. The local minimum conditions for $$ x \in \mathbb{R}^n, \lambda \in \mathbb{R}^m $$ are written as

$$
\begin{split}
& \nabla_x L(x^*, \lambda^*) = 0 \\
& \nabla_\lambda L(x^*, \lambda^*) = 0 \\
& \langle y , \nabla^2_{xx} L(x^*, \lambda^*) y \rangle \geq 0, \;\;\; \forall y \in \mathbb{R}^n : \nabla h(x^*)^\top y = 0
\end{split}
$$

Depending on the behavior of the Hessian, the critical points can have a different character.

![](../kkt_images/critical.png)

## Optimization with inequality conditions
### Example

$$
f(x) = x_1^2 + x_2^2 \;\;\;\; g(x) = x_1^2 + x_2^2 - 1
$$

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
\text{s.t. } & g(x) \leq 0
\end{split}
$$

![](../kkt_images/KKT_p027.png)

![](../kkt_images/KKT_p028.png)

![](../kkt_images/KKT_p029.png)

![](../kkt_images/KKT_p030.png)

Thus, if the constraints of the type of inequalities are inactive in the UM problem, then don't worry and write out the solution to the UM problem. However, this  is not a heal-all :) Consider the second childish example
$$
f(x) = (x_1 - 1.1)^2 + (x_2 + 1.1)^2 \;\;\;\; g(x) = x_1^2 + x_2^2 - 1
$$

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
\text{s.t. } & g(x) \leq 0
\end{split}
$$

![](../kkt_images/KKT_p033.png)

![](../kkt_images/KKT_p034.png)

![](../kkt_images/KKT_p035.png)

![](../kkt_images/KKT_p036.png)

![](../kkt_images/KKT_p037.png)

![](../kkt_images/KKT_p038.png)

![](../kkt_images/KKT_p039.png)

![](../kkt_images/KKT_p040.png)

So, we have a problem:

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
\text{s.t. } & g(x) \leq 0
\end{split}
$$

Two possible cases:

1.
	$$
	\begin{split}
    & g(x^*) < 0 \\
    & \nabla f(x^*) = 0 \\
    & \nabla^2 f(x^*) > 0
    \end{split}
    $$
    
2.
	$$ \begin{split}
    & g(x^*) = 0 \\
    & - \nabla f(x^*) = \mu \nabla g(x^*), \;\; \mu > 0 \\
    & \langle y , \nabla^2_{xx} L(x^*, \mu^*) y \rangle \geq 0, \;\;\; \forall y \in \mathbb{R}^n : \nabla g(x^*)^\top y = 0
    \end{split}
    $$
   

Combining two possible cases, we can write down the general conditions for the problem:

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
\text{s.t. } & g(x) \leq 0 \\
&
\end{split}
$$

Let's define the Lagrange function:

$$
L (x, \mu) = f(x) + \mu g(x)
$$

Then $$x^*$$ point - local minimum of the problem described above, if and only if:

$$
\begin{split}
    & (1) \; \nabla_x L (x^*, \mu^*) = 0 \\
    & (2) \; \mu^* \geq 0 \\
    & (3) \; \mu^* g(x^*) = 0 \\
    & (4) \; g(x^*) \leq 0\\
    & (5) \; \langle y , \nabla^2_{xx} L(x^*, \mu^*) y \rangle \geq 0, \;\;\; \forall y \in \mathbb{R}^n : \nabla g(x^*)^\top y = 0
\end{split}
$$

It's noticeable, that $$L(x^*, \mu^*) = f(x^*)$$. Conditions $$\mu^* = 0 , (1), (4)$$ are the first scenario realization, and conditions $$\mu^* > 0 , (1), (3)$$ - the second.

### General formulation

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n}\\
\text{s.t. } & g_i(x) \leq 0, \; i = 1,\ldots,m\\
& h_j(x) = 0, \; j = 1,\ldots, p
\end{split}
$$

This formulation is a general problem of mathematical programming. From now, we only consider $$ \textbf{regular} $$ tasks. This is a very important remark from a formal point of view. Those wishing to understand in more detail, please refer to Google.

Solution

$$
L(x, \mu, \lambda) = f(x) + \sum\limits_{j=1}^p\lambda_j h_j(x) + \sum\limits_{i=1}^m \mu_i g_i(x)
$$

# Karush-Kuhn-Tucker conditions
{% include tabs.html bibtex = '@misc{kuhn1951nonlinear,
  title={Nonlinear programming, in (J. Neyman, ed.) Proceedings of the Second Berkeley Symposium on Mathematical Statistics and Probability},
  author={Kuhn, Harold W and Tucker, Albert W},
  year={1951},
  publisher={University of California Press, Berkeley}
}' file='/assets/files/kuhntucker.pdf' inline = 'True'%}

{% include tabs.html bibtex = '@article{karush1939minima,
  title={Minima of functions of several variables with inequalities as side constraints},
  author={Karush, William},
  journal={M. Sc. Dissertation. Dept. of Mathematics, Univ. of Chicago},
  year={1939}
}' file='/assets/files/karush.pdf' inline = 'True'%}

Let $$ x^* $$ be a solution to a mathematical programming problem, and the functions $$ f, h_j, g_i $$ are differentiable.
Then there are $$ \lambda^* $$ and $$ \mu^* $$ such that the following conditions are carried out:

* \$$\nabla_x L(x^*, \lambda^*, \mu^*) = 0$$
* \$$\nabla_\lambda L(x^*, \lambda^*, \mu^*) = 0$$
* \$$ \mu^*_j \geq 0$$
* \$$\mu^*_j g_j(x^*) = 0$$
* \$$g_j(x^*) \leq 0$$

These conditions are sufficient if the problem is regular, i.e. if:
1) the given problem is a convex optimization problem (i.e., the functions $$ f $$ and $$ g_i $$ are convex, $$ h_i $$ are affine) and the Slater condition is satisfied;
   or
2) strong duality is fulfilled.

# References
* [Lecture](http://www.csc.kth.se/utbildning/kth/kurser/DD3364/Lectures/KKT.pdf) on KKT conditions (very intuitive explanation) in course "Elements of Statistical Learning" @ KTH.
* [One-line proof of KKT](https://link.springer.com/content/pdf/10.1007%2Fs11590-008-0096-3.pdf)
