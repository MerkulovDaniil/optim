---
layout: default
title: Linear least squares
parent: Applications
---

# Problem
In a least-squares, or linear regression, problem, we have measurements $$ X \in \mathbb{R}^{m \times n} $$ and $$ y \in \mathbb{R}^{m} $$ and seek a vector $$ \theta \in \mathbb{R}^{n} $$ such that $$ X \theta $$ is close to $$ y $$. Closeness is defined as the sum of the squared differences: 

$$ 
\sum\limits_{i=1}^m (x_i^\top \theta - y_i)^2
$$

also known as the $$ l_2 $$-norm squared, $$ \|X \theta - y\|^2_2 $$

For example, we might have a dataset of $m$ users, each represented by $n$ features. Each row $x_i^\top$ of $X$ is the features for user $$ i $$, while the corresponding entry $y_i$ of $y$ is the measurement we want to predict from $x_i^\top$, such as ad spending. The prediction is given by $$ x_i^\top \theta $$.

We find the optimal $\theta$ by solving the optimization problem

$$
\|X \theta - y\|^2_2 \to \min_{\theta \in \mathbb{R}^{n}}
$$

Let $$\theta^*$$ denote the optimal $$ \theta $$. The quantity $$ r=X \theta^* - y $$ is known as the residual. If $$ \|r\|_2 = 0 $$, we have a perfect fit.

Note, that the function needn't be linear in the argument $$x$$ but only in the parameters $$\theta$$ that are to be determined in the best fit.
![](../non_linear_fit.svg)

# Approaches

* If the matrix $$X$$ is relatively small, we can write down and calculate exact solution:

$$
\theta^* = (X^\top X)^{-1} X^\top y
$$

However, this approach squares the condition number of the problem, which could be an obstacle in case of ill-conditioned huge scale problem. 

* Otherwise, we could use iterative methods.

# Code
* [Colab notebook](https://colab.research.google.com/drive/1en8JLreLD4t4SUgzgxB7GyQ7y_fe8Z-X)

# References
* [CVXPY documentation](https://www.cvxpy.org/examples/basic/least_squares.html)
* [Interactive example](http://setosa.io/ev/ordinary-least-squares-regression/)