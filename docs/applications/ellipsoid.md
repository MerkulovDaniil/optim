---
layout: default
title: Minimum volume ellipsoid
parent: Applications
---

# Problem
![](../ellipsoid.svg)

Let $$x_1, \ldots, x_n$$ be the points in $$\mathbb{R}^2$$. Given these points we need to find an ellipsoid, that contains all points with the minimum volume (in 2d case volume of an ellipsoid is just the square).

An invertible linear transformation applied to a unit sphere produces an ellipsoid with the square, that is $$\det A^{-1}$$ times bigger, than the unit sphere square, that's why we parametrize the interior of ellipsoid in the following way:

$$
S = \{x \in \mathbb{R}^2 \; | \; u = Ax + b, \|u\|_2^2 \leq 1\}
$$

Sadly, the determinant is the function, which is relatively hard to minimize explicitly. However, the function $$\log \det A^{-1} = -\log \det A$$ is actually convex, which provides a great opportunity to work with it. As soon as we need to cover all the points with ellipsoid of minimum volume, we pose an optimization problem on the convex function with convex restrictions:


$$
\begin{align*}
& \min_{A \in \mathbb{R}^{2 \times 2}, b \in \mathbb{R}^{2}} -\log\det(A)\\
\text{s.t. } & \|Ax_i + b\| \leq 1, i = 1, \ldots, n\\
& A \succ 0
\end{align*}
$$

![](../ellipsoid2.svg)

# Code
[Open In Colab]( https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Ellipsoid.ipynb){: .btn }

# References
* [Jupyter notebook](https://colab.research.google.com/github/amkatrutsa/MIPT-Opt/blob/master/01-Intro/demos.ipynb#scrollTo=W264L1t1p3mF) by A. Katrutsa
* [https://cvxopt.org/examples/book/ellipsoids.html](CVXOPT documentation)
