# Duality


## Duality

1.  **Toy example**

    $$
     \begin{split}
     & x^2 + 1 \to \min\limits_{x \in \mathbb{R} }\\
     \text{s.t. } & (x-2)(x-4) \leq 0
     \end{split}
     $$

    1.  Give the feasible set, the optimal value, and the optimal
        solution.
    2.  Plot the objective $x^2 +1$ versus $x$. On the same plot, show
        the feasible set, optimal point and value, and plot the
        Lagrangian $L(x,\mu)$ versus $x$ for a few positive values of
        $\mu$. Verify the lower bound property
        ($p^* \geq \inf_x L(x, \mu)$for $\mu \geq 0$). Derive and sketch
        the Lagrange dual function $g$.
    3.  State the dual problem, and verify that it is a concave
        maximization problem. Find the dual optimal value and dual
        optimal solution $\mu^*$. Does strong duality hold?
    4.  Let $p^*(u)$ denote the optimal value of the problem

    $$
     \begin{split}
     & x^2 + 1 \to \min\limits_{x \in \mathbb{R} }\\
     \text{s.t. } & (x-2)(x-4) \leq u
     \end{split}
     $$

    as a function of the parameter $u$. Plot $p^*(u)$. Verify that
    $\dfrac{dp^*(0)}{du} = -\mu^*$

2.  **Dual vs conjugate.** Consider the following optimization problem

    $$
     \begin{split}
     & f(x) \to \min\limits_{x \in \mathbb{R}^{n} }\\
     \text{s.t. } & x = 0 \\
     \end{split}
     $$

    1.  Find Lagrangian of the primal problem
    2.  Find the dual function
    3.  Write down the dual problem

3.  **Dual vs conjugate.**Consider the following optimization problem

    $$
     \begin{split}
     & f(x) \to \min\limits_{x \in \mathbb{R}^{n} }\\
     \text{s.t. } & Ax \preceq b \\
     & Cx = d \\
     \end{split}
     $$

    1.  Find Lagrangian of the primal problem
    2.  Find the dual function
    3.  Write down the dual problem

4.  **Dual vs conjugate.** Consider the following optimization problem

    $$
     \begin{split}
     & f_0(x) = \sum\limits_{i=1}^n f_i(x_i) \to \min\limits_{x \in \mathbb{R}^{n} }\\
     \text{s.t. } & a^\top x = b \\
     & f_i(x) - \text{ differentiable and strictly convex}
     \end{split}
     $$

    1.  Find Lagrangian of the primal problem
    2.  Find the dual function
    3.  Write down the dual problem

5.  **New variables.** Consider an unconstrained problem of the form:

    $$
     f_0(Ax + b) \to \min\limits_{x \in \mathbb{R}^{n} }
     $$

    And its equivalent reformulation:

    $$
     \begin{split}
     & f_0(y) \to \min\limits_{y \in \mathbb{R}^{m} }\\
     \text{s.t. } & y = Ax + b \\
     \end{split}
     $$

    1.  Find Lagrangian of the primal problems
    2.  Find the dual functions
    3.  Write down the dual problems

6.  The weak duality inequality, $d^* ≤ p^*$ , clearly holds when
    $d^* = -\infty$ or $p^* = \infty$. Show that it holds in the other
    two cases as well: If $p^* = −\infty$, then we must have
    $d^* = −\infty$, and also, if $d^* = \infty$, then we must have
    $p^* = \infty.$

7.  Express the dual problem of

    $$
    \begin{split}
    & c^\top x\to \min\limits_{x \in \mathbb{R}^{n} }\\
    \text{s.t. } & f(x) \leq 0
    \end{split}
    $$

    with $c \neq 0$, in terms of the conjugate function $f^*$. Explain
    why the problem you give is convex. We do not assume $f$ is convex.

8.  **Least Squares.** Let we have the primal problem:

    $$
     \begin{split}
     & x^\top x \to \min\limits_{x \in \mathbb{R}^{n} }\\
     \text{s.t. } & Ax = b
     \end{split}
     $$

    1.  Find Lagrangian of the primal problem
    2.  Find the dual function
    3.  Write down the dual problem
    4.  Check whether problem holds strong duality or not
    5.  Write down the solution of the dual problem

9.  **Standard form LP.** Let we have the primal problem:

    $$
     \begin{split}
     & c^\top x \to \min\limits_{x \in \mathbb{R}^{n} }\\
     \text{s.t. } & Ax = b \\
     & x \succeq 0
     \end{split}
     $$

    1.  Find Lagrangian of the primal problem
    2.  Find the dual function
    3.  Write down the dual problem
    4.  Check whether problem holds strong duality or not
    5.  Write down the solution of the dual problem

10. **Two-way partitioning problem.** Let we have the primal problem:

    $$
     \begin{split}
     & x^\top W x \to \min\limits_{x \in \mathbb{R}^{n} }\\
     \text{s.t. } & x_i^2 = 1, i = 1, \ldots, n \\
     \end{split}
     $$

    1.  Find Lagrangian of the primal problem
    2.  Find the dual function
    3.  Write down the dual problem
    4.  Check whether problem holds strong duality or not
    5.  Write down the solution of the dual problem
    6.  Can you reduce this problem to the eigenvalue problem? 🐱

11. **Entropy maximization.** Let we have the primal problem:

    $$
     \begin{split}
     & \sum_i x_i \ln x_i \to \min\limits_{x \in \mathbb{R}^{n} }\\
     \text{s.t. } & Ax \preceq b \\
     & 1^\top x = 1 \\
     & x \succ 0
     \end{split}
     $$

    1.  Find Lagrangian of the primal problem
    2.  Find the dual function
    3.  Write down the dual problem
    4.  Check whether problem holds strong duality or not
    5.  Write down the solution of the dual problem

12. **Minimum volume covering ellipsoid.** Let we have the primal
    problem:

    $$
     \begin{split}
     & \ln \text{det} X^{-1} \to \min\limits_{X \in \mathbb{S}^{n}_{++} }\\
     \text{s.t. } & a_i^\top X a_i \leq 1 , i = 1, \ldots, m
     \end{split}
     $$

    1.  Find Lagrangian of the primal problem
    2.  Find the dual function
    3.  Write down the dual problem
    4.  Check whether problem holds strong duality or not
    5.  Write down the solution of the dual problem

13. **Equality constrained norm minimization.** Let we have the primal
    problem:

    $$
     \begin{split}
     & \|x\| \to \min\limits_{x \in \mathbb{R}^{n} }\\
     \text{s.t. } & Ax = b
     \end{split}
     $$

    1.  Find Lagrangian of the primal problem
    2.  Find the dual function
    3.  Write down the dual problem
    4.  Check whether problem holds strong duality or not
    5.  Write down the solution of the dual problem

14. **Inequality form LP.** Let we have the primal problem:

    $$
     \begin{split}
     & c^\top x \to \min\limits_{x \in \mathbb{R}^{n} }\\
     \text{s.t. } & Ax \preceq b \\
     & x \succeq 0
     \end{split}
     $$

    1.  Find Lagrangian of the primal problem
    2.  Find the dual function
    3.  Write down the dual problem
    4.  Check whether problem holds strong duality or not
    5.  Write down the solution of the dual problem

15. **Nonconvex strong duality** Let we have the primal problem:

    $$
     \begin{split}
     & x^\top Ax +2b^\top x \to \min\limits_{x \in \mathbb{R}^{n} }\\
     \text{s.t. } & x^\top x \leq 1 \\
     & A \in \mathbb{S}^n, A \nsucceq 0, b \in \mathbb{R}^n
     \end{split}
     $$

    1.  Find Lagrangian of the primal problem
    2.  Find the dual function
    3.  Write down the dual problem
    4.  Check whether problem holds strong duality or not
    5.  Write down the solution of the dual problem

16. **A penalty method for equality constraints.** We consider the
    problem minimize

    $$
     \begin{split}
     & f_0(x) \to \min\limits_{x \in \mathbb{R}^{n} }\\
     \text{s.t. } & Ax = b,
     \end{split}
     $$

    where \$f_0(x): ^n \$ is convex and differentiable, and
    $A \in \mathbb{R}^{m \times n}$ with $\mathbf{rank }A = m$. In a
    quadratic penalty method, we form an auxiliary function

    $$
     \phi(x) = f_0(x) + \alpha \|Ax - b\|_2^2,
     $$

    where $\alpha > 0$ is a parameter. This auxiliary function consists
    of the objective plus the penalty term $\alpha \|Ax - b\|_2^2$. The
    idea is that a minimizer of the auxiliary function, $\tilde{x}$,
    should be an approximate solution of the original problem. Intuition
    suggests that the larger the penalty weight $\alpha$, the better the
    approximation $\tilde{x}$ to a solution of the original problem.
    Suppose $\tilde{x}$ is a minimizer of $\phi(x)$. Show how to find,
    from $\tilde{x}$, a dual feasible point for the original problem.
    Find the corresponding lower bound on the optimal value of the
    original problem.

17. **Analytic centering.** Derive a dual problem for

    $$
     -\sum_{i=1}^m \log (b_i - a_i^\top x) \to \min\limits_{x \in \mathbb{R}^{n} }
     $$

    with domain $\{x \mid a^\top_i x < b_i , i = [ 1,m ] \}$. First
    introduce new variables $y_i$ and equality constraints
    $y_i = b_i − a^\top_i x$. (The solution of this problem is called
    the analytic center of the linear inequalities
    $a^\top_i x \leq b_i ,i = [ 1,m ]$. Analytic centers have geometric
    applications, and play an important role in barrier methods.)
