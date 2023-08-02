---
layout: default
title: Natural gradient descent
parent: Adaptive metric methods
grand_parent: Methods
nav: 3
bibtex: |
  @article{amari1998natural,
  title={Natural gradient works efficiently in learning},
  author={Amari, Shun-Ichi},
  journal={Neural computation},
  volume={10},
  number={2},
  pages={251--276},
  year={1998},
  publisher={MIT Press}
  }
file: /assets/files/NGD.pdf
---
# Intuition
Let's consider illustrative example of a simple function of 2 variables:

$$
f(x_1, x_2) = 2x_1 + \frac{1}{3}x_2, \quad \nabla_x f = \begin{pmatrix} 2\\ \frac{1}{3} \end{pmatrix}
$$

Now, let's introduce new variables $$(y_1, y_2) = (2x_1, \frac{1}{3}x_2) $$ or $$y = Bx$$, where $$B = \begin{pmatrix} 2 & 0\\ 0 & \frac{1}{3} \end{pmatrix}$$. The same function, written in the new coordinates, is

$$
f(y_1, y_2) = y_1 + y_2, \quad \nabla_y f = \begin{pmatrix} 1\\ 1 \end{pmatrix}
$$

Let’s summarize what happened:
* We have a transformation of a vector space described by a coordinate transformation matrix B.
* Coordinate vectors transforms as $$y = Bx$$.
* However, the partial gradient of a function w.r.t. the
coordinates transforms as $$\frac{\partial f}{\partial y} = B^{-\top} \frac{\partial f}{\partial x}$$.
* Therefore, there seems to exist one type of mathematical objects (e.g. coordinate vectors) which transform with $$B$$, and a second type of mathematical objects (e.g. the partial gradient of a function w.r.t. coordinates) which transform with $$B^{-\top}$$.

These two types are called *contra-variant* and *co-variant*. This should at least tell us that indeed the so-called “gradient-vector” is somewhat different to a “normal vector”: it behaves inversely under coordinate transformations.

Nice thing here is that steepest descent direction $$A_x^{-1}\nabla_x f$$ on a sphere transforms as a covariant vector, since $$A_y = B^{-\top} A_x B^{-1}$$:

$$
\begin{split}
A_y^{-1}\nabla_y f = \\
(B^{-\top} A_x B^{-1})^{-1} B^{-\top} \nabla_x f = \\
B A_x^{-1} B^\top B^{-\top} \nabla_x f = \\
B (A_x^{-1} \nabla_x f)
\end{split}
$$

# Steepest descent in distribution space

Suppose, we have a probabilistic model represented by its likelihood $$p(x \vert \theta) $$. We want to maximize this likelihood function to find the most likely parameter $$\theta$$ with given observations. Equivalent formulation would be to minimize the loss function $$\mathcal{L}(\theta)$$, which is the negative logarithm of likelihood function.

# Example


# References
* [Some notes on gradient descent](https://ipvs.informatik.uni-stuttgart.de/mlr/marc/notes/gradientDescent.pdf)
* [Natural Gradient Descent](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/)

# Code
[Open In Colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/NGD.ipynb){: .btn }