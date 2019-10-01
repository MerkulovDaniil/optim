---
layout: default
title: Linear least squares
parent: Applications
---

# Problem

![](../lls_idea.svg)
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

## Moore–Penrose inverse
If the matrix $$X$$ is relatively small, we can write down and calculate exact solution:

$$
\theta^* = (X^\top X)^{-1} X^\top y = X^\dagger y, 
$$

where $$X^\dagger$$ is called [pseudo-inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) matrix. However, this approach squares the condition number of the problem, which could be an obstacle in case of ill-conditioned huge scale problem. 

## QR decomposition
For any matrix $$X \in \mathbb{R}^{m \times n}$$ there is exists QR decomposition:

$$
X = Q \cdot R,
$$

where  $$Q$$ is an orthogonal matrix (its columns are orthogonal unit vectors meaning  $$Q^\top Q=QQ^\top=I$$ and $$R$$ is an upper triangular matrix. It is important to notice, that since $$Q^{-1} = Q^\top$$, we have:

$$
QR\theta = y \quad \longrightarrow \quad R \theta = Q^\top y
$$

Now, process of finding theta consists of two steps:
1. Find the QR decomposition of $$X$$.
1. Solve triangular system $$R \theta = Q^\top y$$, which is triangular and, therefore, easy to solve.

## Cholesky decomposition
For any positive definite matrix $$A \in \mathbb{R}^{n \times n}$$ there is exists Cholesky decomposition:

$$
X^\top X = A = L^\top \cdot L,
$$

where  $$L$$ is an lower triangular matrix. We have:

$$
L^\top L\theta = y \quad \longrightarrow \quad L^\top z_\theta = y
$$

Now, process of finding theta consists of two steps:
1. Find the Cholesky decomposition of $$X^\top X$$.
1. Find the $$z_\theta = L\theta$$ by solving triangular system $$L^\top z_\theta = y$$
1. Find the $$\theta$$ by solving triangular system $$L\theta = z_\theta$$

Note, that in this case the error stil proportional to the squared condition number.

![](../lls_times.svg)


# Code
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Least_squares.ipynb)
# References
* [CVXPY documentation](https://www.cvxpy.org/examples/basic/least_squares.html)
* [Interactive example](http://setosa.io/ev/ordinary-least-squares-regression/)
* [Jupyter notebook by A. Katrutsa](https://nbviewer.jupyter.org/github/amkatrutsa/MIPT-Opt/blob/master/16-LSQ/Seminar16en.ipynb)