# Quasi Newton methods


## Intuition

For the classic task of unconditional optimization
$f(x) \to \min\limits_{x \in \mathbb{R}^n}$ the general scheme of
iteration method is written as:

$$
x_{k+1} = x_k + \alpha_k s_k
$$

In the Newton method, the $s_k$ direction (Newton’s direction) is set by
the linear system solution at each step:

$$
s_k = - B_k\nabla f(x_k), \;\;\; B_k = f_{xx}^{-1}(x_k)
$$

i.e. at each iteration it is necessary to **compensate** hessian and
gradient and **resolve** linear system.

Note here that if we take a single matrix of $B_k = I_n$ as $B_k$ at
each step, we will exactly get the gradient descent method.

The general scheme of quasi-Newton methods is based on the selection of
the $B_k$ matrix so that it tends in some sense at $k \to \infty$ to the
true value of inverted Hessian in the local optimum $f_{xx}^{-1}(x_*)$.
Let’s consider several schemes using iterative updating of $B_k$ matrix
in the following way:

$$
B_{k+1} = B_k + \Delta B_k
$$

Then if we use Taylor’s approximation for the first order gradient, we
get it:

$$
\nabla f(x_k) - \nabla f(x_{k+1}) \approx f_{xx}(x_{k+1}) (x_k - x_{k+1}).
$$

Now let’s formulate our method as:

$$
\Delta x_k = B_{k+1} \Delta y_k, \text{ where } \;\; \Delta y_k = \nabla f(x_{k+1}) - \nabla f(x_k)
$$

in case you set the task of finding an update $\Delta B_k$:

$$
\Delta B_k \Delta y_k = \Delta x_k - B_k \Delta y_k
$$

## Broyden method

The simplest option is when the amendment $\Delta B_k$ has a rank equal
to one. Then you can look for an amendment in the form

$$
\Delta B_k = \mu_k q_k q_k^\top.
$$

where $\mu_k$ is a scalar and $q_k$ is a non-zero vector. Then mark the
right side of the equation to find $\Delta B_k$ for $\Delta z_k$:

$$
\Delta z_k = \Delta x_k - B_k \Delta y_k
$$

We get it:

$$
\mu_k q_k q_k^\top \Delta y_k = \Delta z_k
$$

$$
\left(\mu_k \cdot q_k^\top \Delta y_k\right) q_k = \Delta z_k
$$

A possible solution is: $q_k = \Delta z_k$,
$\mu_k = \left(q_k^\top \Delta y_k\right)^{-1}$.

Then an iterative amendment to Hessian’s evaluation at each iteration:

$$
\Delta B_k = \dfrac{(\Delta x_k - B_k \Delta y_k)(\Delta x_k - B_k \Delta y_k)^\top}{\langle \Delta x_k - B_k \Delta y_k , \Delta y_k\rangle}.
$$

## Davidon–Fletcher–Powell method

$$
\Delta B_k = \mu_1 \Delta x_k (\Delta x_k)^\top + \mu_2 B_k \Delta y_k (B_k \Delta y_k)^\top.
$$

$$
\Delta B_k = \dfrac{(\Delta x_k)(\Delta x_k )^\top}{\langle \Delta x_k , \Delta y_k\rangle} - \dfrac{(B_k \Delta y_k)( B_k \Delta y_k)^\top}{\langle B_k \Delta y_k , \Delta y_k\rangle}.
$$

## Broyden–Fletcher–Goldfarb–Shanno method

$$
\Delta B_k = Q U Q^\top, \quad Q = [q_1, q_2], \quad q_1, q_2 \in \mathbb{R}^n, \quad U = \begin{pmatrix} a & c\\ c & b \end{pmatrix}.
$$

$$
\Delta B_k = \dfrac{(\Delta x_k)(\Delta x_k )^\top}{\langle \Delta x_k , \Delta y_k\rangle} - \dfrac{(B_k \Delta y_k)( B_k \Delta y_k)^\top}{\langle B_k \Delta y_k , \Delta y_k\rangle} + p_k p_k^\top. 
$$

## Code

- [Open In
  Colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Quasi_Newton.ipynb)
- [Comparison of quasi Newton
  methods](https://nbviewer.jupyter.org/github/fabianp/pytron/blob/master/doc/benchmark_logistic.ipynb)
