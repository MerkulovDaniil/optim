---
layout: default
parent: Theory
title: Matrix calculus
nav_order: 1
---

# Useful definitions and notations
We will treat all vectors as column vectors by default.
## Matrix and vector multiplication
Let $$A$$ be $$m \times n$$, and $$B$$ be $$n \times p$$, and let the product $$AB$$ be:

$$
C = AB
$$

then $$C$$ is a $$m \times p$$ matrix, with element $(i, j)$ given by: 

$$
c_{ij} = \sum_{k=1}^n a_{ik}b_{kj}
$$

Let $$A$$ be $$m \times n$$, and $$x$$ be $$n \times 1$$, then the typical element of the product:

$$
z = Ax
$$

is given by:

$$
z_i = \sum_{k=1}^n a_{ik}x_k
$$

Finally, just to remind:

* \$$C = AB \quad C^\top = B^\top A^\top$$
* \$$AB \neq BA$$
* \$$e^{A} =\sum\limits_{k=0}^{\infty }{1 \over k!}A^{k}$$
* $$e^{A+B} \neq e^{A} e^{B}$$ (but if $$A$$ and $$B$$ are commuting matrices, which means that $$AB = BA$$, $$e^{A+B} = e^{A} e^{B}$$)
* \$$\langle x, Ay\rangle = \langle A^\top x, y\rangle$$

## Gradient
Let  $$f(x):\mathbb{R}^n→\mathbb{R}$$, then vector, which contains all first order partial derivatives:

$$
\nabla f(x) = \dfrac{df}{dx} = \begin{pmatrix}
    \frac{\partial f}{\partial x_1} \\
    \frac{\partial f}{\partial x_2} \\
    \vdots \\
    \frac{\partial f}{\partial x_n}
\end{pmatrix}
$$

## Hessian 
Let  $$f(x):\mathbb{R}^n→\mathbb{R}$$, then matrix, containing all the second order partial derivatives:

$$
f''(x) = \dfrac{\partial^2 f}{\partial x_i \partial x_j} = \begin{pmatrix}
    \frac{\partial^2 f}{\partial x_1 \partial x_1} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \dots  & \frac{\partial^2 f}{\partial x_1\partial x_n} \\
    \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2 \partial x_2} & \dots  & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \dots  & \frac{\partial^2 f}{\partial x_n \partial x_n}
\end{pmatrix}
$$

But actually, Hessian could be a tensor in such a way: $$\left(f(x): \mathbb{R}^n \to \mathbb{R}^m \right)$$ is just 3d tensor, every slice is just hessian of corresponding scalar function $$\left( H\left(f_1(x)\right), H\left(f_2(x)\right), \ldots, H\left(f_m(x)\right)\right)$$.

## Jacobian
The extension of the gradient of multidimensional  $$f(x):\mathbb{R}^n→\mathbb{R}^m$$ :

$$
f'(x) = \dfrac{df}{dx^T} = \begin{pmatrix}
    \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \dots  & \frac{\partial f_1}{\partial x_n} \\
    \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \dots  & \frac{\partial f_2}{\partial x_n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \dots  & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}
$$

## Summary


$$
f(x) : X \to Y; \;\;\;\;\;\;\;\; \frac{\partial f(x)}{\partial x} \in G
$$

|             X             |        Y       |             G             |                   Name                  |
|:-------------------------:|:--------------:|:-------------------------:|:----------------------------------------------:|
|        $\mathbb{R}$       |  $\mathbb{R}$  |        $\mathbb{R}$       |              $f'(x)$ (derivative)             |
|       $\mathbb{R}^n$      |  $\mathbb{R}$  |       $\mathbb{R^n}$      |  $\dfrac{\partial f}{\partial x_i}$ (gradient) |
|       $\mathbb{R}^n$      | $\mathbb{R}^m$ | $\mathbb{R}^{m \times n}$ | $\dfrac{\partial f_i}{\partial x_j}$ (jacobian) |
| $\mathbb{R}^{m \times n}$ |  $\mathbb{R}$  | $\mathbb{R}^{m \times n}$ |      $\dfrac{\partial f}{\partial x_{ij}}$     |

named gradient of  $$f(x)$$. This vector indicates the direction of steepest ascent. Thus, vector  $$−\nabla f(x)$$  means the direction of the steepest descent of the function in the point. Moreover, the gradient vector is always orthogonal to the contour line in the point.

# General concept
## Naive approach
The basic idea of naive approach is to reduce matrix/vector derivatives to the well-known scalar derivatives.
![](../matrix_calculus.svg)
One of the most important practical tricks here is to separate indices of sum ($$i$$) and partial derivatives ($$k$$). Ignoring this simple rule tends to produce mistakes.
## Guru approach
The guru approach implies formulating a set of simple rules, which allows you to calculate derivatives just like in a scalar case. It might be convenient to use the differential notation here.

### Differentials
After obtaining the differential notation of $df$ we can retrieve the gradient using following formula:

$$
df(x) = \langle \nabla f(x), dx\rangle
$$

Then, if we have differential of the above form and we need to calculate the second derivative of the matrix/vector function, we treat "old" $$dx$$ as the constant $$dx_1$$, then calculate $$d(df)$$

$$
d^2f(x) = \langle \nabla^2 f(x) dx_1, dx_2\rangle = \langle H_f(x) dx_1, dx_2\rangle
$$

### Properties

Let $$A$$ and $$B$$ be the constant matrices, while $$X$$ and $$Y$$ are the variables (or matrix functions).

* \$$dA = 0$$
* \$$d(\alpha X) = \alpha (dX)$$
* \$$d(AXB) = A(dX )B$$
* \$$d(X+Y) = dX + dY$$
* \$$d(X^\top) = (dX)^\top$$
* \$$d(XY) = (dX)Y + X(dY)$$
* \$$d\langle X, Y\rangle = \langle dX, Y\rangle+ \langle X, dY\rangle$$
* \$$d\left( \dfrac{X}{\phi}\right) = \dfrac{\phi dX - (d\phi) X}{\phi^2}$$
* \$$d\left( \det X \right) = \det X \langle X^{-\top}, dX \rangle $$
* \$$d\left(\text{tr } X \right) = \langle I, dX\rangle$$
* \$$df(g(x)) = \dfrac{df}{dg} \cdot dg(x)$$
* \$$H = (J(\nabla f))^T$$
* \$$ d(X^{-1})=-X^{-1}(dX)X^{-1}$$


# References
* [Good introduction](https://atmos.washington.edu/~dennis/MatrixCalculus.pdf)
* [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
* [MSU seminars](http://www.machinelearning.ru/wiki/images/a/ab/MOMO18_Seminar1.pdf) (Rus.)
* [Online tool](http://www.matrixcalculus.org/) for analytic expression of a derivative.
* [Determinant derivative](https://charlesfrye.github.io/math/2019/01/25/frechet-determinant.html)
