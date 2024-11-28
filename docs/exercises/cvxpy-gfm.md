# CVXPY library


## CVXPY library

1.  **Constrained linear least squares** Solve the following problem
    with cvxpy library.

    $$
     \begin{split} &\|X \theta - y\|^2_2 \to \min\limits_{\theta \in \mathbb{R}^{n} } \\ \text{s.t. } & 0_n \leq \theta \leq 1_n \end{split}
     $$

2.  **Linear programming** A linear program is an optimization problem
    with a linear objective and affine inequality constraints. A common
    standard form is the following:

    $$  
         \begin{array}{ll}
         \text{minimize}   & c^Tx \\
         \text{subject to} & Ax \leq b.
         \end{array}
     $$

    Here $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$, and
    $c \in \mathbb{R}^n$ are problem data and $x \in \mathbb{R}^{n}$ is
    the optimization variable. The inequality constraint $Ax \leq b$ is
    elementwise. Solve this problem with cvxpy library.

3.  List the installed solvers in cvxpy using `cp.installed_solvers()`
    method.

4.  Solve the following optimization problem using CVXPY:

    $$
     \begin{array}{ll} 
     \text{minimize} & |x| - 2\sqrt{y}\\
     \text{subject to} & 2 \geq e^x \\
     & x + y = 5,
     \end{array}
     $$

    where $x,y \in \mathbb{R}$ are variables. Find the optimal values of
    $x$ and $y$.

5.  **Risk budget allocation** Suppose an amount $x_i>0$ is invested in
    $n$ assets, labeled $i=1,..., n$, with asset return covariance
    matrix $\Sigma \in \mathcal{S}_{++}^n$. We define the *risk* of the
    investments as the standard deviation of the total return

    $$
     R(x) = (x^T\Sigma x)^{1/2}.
     $$

    We define the (relative) *risk contribution* of asset $i$ (in the
    portfolio $x$) as

    $$
     \rho_i = \frac{\partial \log R(x)}{\partial \log x_i} =
     \frac{\partial R(x)}{R(x)} \frac{x_i}{\partial x_i}, \quad i=1, \ldots, n.
     $$

    Thus $\rho_i$ gives the fractional increase in risk per fractional
    increase in investment $i$. We can express the risk contributions as

    $$
     \rho_i = \frac{x_i (\Sigma x)_i} {x^T\Sigma x}, \quad i=1, \ldots, n,
     $$

    from which we see that $\sum_{i=1}^n \rho_i = 1$. For general $x$,
    we can have $\rho_i <0$, which means that a small increase in
    investment $i$ decreases the risk. Desirable investment choices have
    $\rho_i>0$, in which case we can interpret $\rho_i$ as the fraction
    of the total risk contributed by the investment in asset $i$. Note
    that the risk contributions are homogeneous, i.e., scaling $x$ by a
    positive constant does not affect $\rho_i$.

    - **Problem statement:** In the *risk budget allocation problem*, we
      are given $\Sigma$ and a set of desired risk contributions
      $\rho_i^\mathrm{des}>0$ with $\bf{1}^T \rho^\mathrm{des}=1$; the
      goal is to find an investment mix $x\succ 0$, $\bf{1}^Tx =1$, with
      these risk contributions. When $\rho^\mathrm{des} = (1/n)\bf{1}$,
      the problem is to find an investment mix that achieves so-called
      *risk parity*.

    - **a)** Explain how to solve the risk budget allocation problem
      using convex optimization. *Hint.* Minimize
      $(1/2)x^T\Sigma x - \sum_{i=1}^n \rho_i^\mathrm{des} \log x_i$.

    - **b)** Find the investment mix that achieves risk parity for the
      return covariance matrix $\Sigma$ below.

      ``` python
      import numpy as np
      import cvxpy as cp
      Sigma = np.array(np.matrix("""6.1  2.9  -0.8  0.1;
                           2.9  4.3  -0.3  0.9;
                          -0.8 -0.3   1.2 -0.7;
                           0.1  0.9  -0.7  2.3"""))
      rho = np.ones(4)/4
      ```

## Materials

- [CVXPY
  exercises](https://github.com/cvxgrp/cvx_short_course/tree/master/exercises)
- [Additional Exercises for Convex
  Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook_extra_exercises.pdf)
