---
title: "Multidimensional scaling"
---

:::{.video}
mds.mp4
:::

Why does everyone use gradient methods to train large models?

We'll demonstrate it with the example of solving the Multidimensional Scaling (MDS) problem. In this problem, we need to draw objects on the plane that we only know how far each of them is from each other. That is, we have a matrix of pairwise distances as input, and coordinates of objects on the plane as output. And these coordinates should be such that the ratio of distances between the objects remains as close as possible to the original ones. As a loss function is the sum of squares of deviations of distances between cities at the current coordinates and the given value.

Suppose, we have a pairwise distance matrix for $N$ $d$-dimensional objects $D \in \mathbb{R}^{N \times N}$. Given this matrix, our goal is to recover the initial coordinates $W_i \in \mathbb{R}^d, \; i = 1, \ldots, N$.

$$
L(W) = \sum_{i, j = 1}^N \left(\|W_i - W_j\|^2_2 - D_{i,j}\right)^2 \to \min_{W \in \mathbb{R}^{N \times d}}
$$

Despite the increase in dimensionality, the entire trajectory of the method can be visualized in the plane. The number of variables in the problem here is 2*(number of cities), since for each of the cities we are looking for 2 coordinates in the plane.

️We run in a sequence the usual gradient descent, which requires knowledge of the gradient, and the Nelder-Mead method from scipy (a widely used gradient-free method). At first the problem is solved for the 6 most populated European cities, then for 15 and for 34 (that's all the European cities with millions of residents from wikipedia). 

It can be seen that the more cities on the map (the higher the dimensionality of the problem), the larger the gap between gradient and gradient-free methods. This is one of the main reasons why we need not only the value of the function being minimized, but also its derivative to solve huge-scale problems.

It turns out that for gradient methods (under a set of reasonable assumptions), the number of iterations required before the method converges does not depend directly on the dimensionality of the problem. That is, if you consider a correctly tuned gradient descent on a ten-dimensional problem, it will need, say, at most 20 iterations to converge. And if you take conventionally the same problem but 100500-dimensional, it will need the same 20 iterations. Of course, the cost of one iteration grows with the dimensionality of the problem, but at least the number of iterations does not grow.


[\faPython Code](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/MDS.ipynb)
