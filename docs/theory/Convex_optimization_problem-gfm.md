# Convex optimization problem


## Convex optimization problem

![The idea behind the definition of a convex optimization
problem](cop.svg)

Note, that there is an agreement in notation of mathematical
programming. The problems of the following type are called **Convex
optimization problem**:

$$
\begin{split}
& f_0(x) \to \min\limits_{x \in \mathbb{R}^n}\\
\text{s.t. } & f_i(x) \leq 0, \; i = 1,\ldots,m\\
& Ax = b,
\end{split}
\tag{COP}
$$

where all the functions $f_0(x), f_1(x), \ldots, f_m(x)$ are convex and
all the equality constraints are affine. It sounds a bit strange, but
not all convex problems are convex optimization problems.

$$
\tag{CP}
f_0(x) \to \min\limits_{x \in S},
$$

where $f_0(x)$ is a convex function, defined on the convex set $S$. The
necessity of affine equality constraint is essential.

> [!EXAMPLE]
>
> ### Example
>
> <div>
>
> <div class="callout-example">
>
> This problem is not a convex optimization problem (but implies
> minimizing the convex function over the convex set):
>
> $$
> \begin{split}
> & x_1^2 + x_2^2 \to \min\limits_{x \in \mathbb{R}^n}\\
> \text{s.t. } & \dfrac{x_1}{1 + x_2^2} \leq 0\\
> & (x_1 + x_2)^2 = 0,
> \end{split}
> \tag{CP}
> $$
>
> while the following equivalent problem is a convex optimization
> problem
>
> $$
> \begin{split}
> & x_1^2 + x_2^2 \to \min\limits_{x \in \mathbb{R}^n}\\
> \text{s.t. } & \dfrac{x_1}{1 + x_2^2} \leq 0\\
> & x_1 + x_2 = 0,
> \end{split}
> \tag{COP}
> $$
>
> </div>
>
> </div>

Such confusion in notation is sometimes being avoided by naming problems
of type $\text{(CP)}$ as *abstract form convex optimization problem*.

## Materials

- [Convex Optimization — Boyd & Vandenberghe @
  Stanford](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
