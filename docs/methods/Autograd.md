---
layout: default
parent: Methods
title: Automatic differentiation
nav_order: 6
---

# Idea
![](https://raw.githubusercontent.com/MerkulovDaniil/optim/master/docs/methods/differentiation_scheme.svg)
Automatic differentiation is a scheme, that allow you to compute a value of gradient of function with linear cost.
## Chain rule
We will illustrate some important matrix calculus facts for specific cases
### Univariate chain rule
Suppose, we have the following functions $R: \mathbb{R} \to \mathbb{R} , L: \mathbb{R} \to \mathbb{R}$ and $W \in \mathbb{R}$. Then

$$
\dfrac{\partial R}{\partial W} = \dfrac{\partial R}{\partial L} \dfrac{\partial L}{\partial W}
$$

### Multivariate chain rule

## Backpropagation
The whole idea came from the applying chain rule to the computation graph of primitive operations
$$
L = L\left(y\left(z(w,x,b)\right), t\right) 
$$
![](https://raw.githubusercontent.com/MerkulovDaniil/optim/master/docs/methods/backprop.svg)
$$
\begin{aligned}
&z = wx+b   &\frac{\partial z}{\partial w} =x, \frac{\partial z}{\partial x} =w, \frac{\partial z}{\partial b} =0  \\
&y = \sigma(z) &\frac{\partial y}{\partial z} =\sigma'(z)\\
&L = \dfrac{1}{2}(y-t)^2 &\frac{\partial L}{\partial y} =y-t, \frac{\partial L}{\partial t} = t -y 
\end{aligned}
$$
## Jacobian vector product
## Hessian vector product
# Code


# Materials
