# Line search


## Line search

1.  Which function is called unimodal?

2.  Derive the convergence speed for a dichotomy method for a unimodal
    function. What type of convergence does this method have?

3.  Consider the function
    $f(x) = (x + \sin x) e^x, \;\;\; x \in [-20, 0]$.

    ![Illustration](Unimodal.svg)

    Consider the following modification of solution localization method,
    in which the interval $[a,b]$ is divided into $2$ parts in a fixed
    proportion of $t: x_t = a + t*(b-a)$ (maximum twice at iteration -
    as in the dichotomy method). Experiment with different values of
    $t \in [0,1]$ and plot the dependence of $N (t)$ - the number of
    iterations needed to achieve $\varepsilon$ - accuracy from the $t$
    parameter. Consider $\varepsilon = 10^{-7}$. Note that with
    $t = 0.5$ this method is exactly the same as the dichotomy method.

4.  Describe the idea of successive parabolic interpolation. What type
    of convergence does this method have?

5.  Write down Armijo–Goldstein condition.

6.  Show that if $0 < c_2 < c_1 < 1$, there may be no step lengths that
    satisfy the Wolfe conditions (sufficient decrease and curvature
    condition).

7.  Show that the one-dimensional minimizer of a strongly convex
    quadratic function always satisfies the Goldstein conditions.

8.  Consider the Rosenbrock function:

    $$
     f(x_1, x_2) =  10(x_2 − x_1^2)^2 + (x_1 − 1)^22
     $$

    You are given the starting point $x_0 = (-1, 2)^\top$. Implement the
    gradient descent algorithm:

    $$
     x^{k+1} = x^k - \alpha^k \nabla f(x^k),
     $$

    where the stepsize is choosen at each iteration via solution of the
    following line search problem

    $$
     \alpha^k = \arg\min\limits_{\alpha \in \mathbb{R}^+}{f(x^k - \alpha \nabla f(x^k))}.
     $$

    Implement any line search method in this problem and plot 2 graphs:
    function value from iteration number and function value from the
    number of function calls (calculate only the function calls, don’t
    include the gradient calls).

9.  Consider the function
    $f(x) = (x + \sin x) e^x, \;\;\; x \in [-20, 0]$
    ![Illustration](https://raw.githubusercontent.com/MerkulovDaniil/optim/master/docs/exercises/Unimodal.svg)

    1.  Implement golden search and binary search methods for this
        function.

    2.  Minimize the function with these two methods and add Brent
        method from
        [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brent.html).
        Compare 3 methods in terms of iterations, time, number of oracle
        calls.
