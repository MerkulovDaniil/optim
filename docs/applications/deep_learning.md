---
layout: default
title: Deep learning
parent: Applications
---

# Problem

![](../dl.png)

A lot of practical task nowadays are being solved by the deep learning approach, which is usually implies finding local minimum of a non - convex function, that generalizes well (enough 😉). The goal of this short text is to provide you an importance of the optimization behind neural network training.

## Cross entropy
One of the most commonly used loss functions in classification tasks is the normalized categorical cross entropy in $$K$$ class problem:

$$
L(\theta) = - \dfrac{1}{n}\sum_{i=1}^n (y_i^\top\log(h_\theta(x_i)) + (1 - y_i)^\top\log(1 - h_\theta(x_i))), \qquad h_\theta^k(x_i) = \dfrac{e^{\theta_k^\top x_i}}{\sum_{j = 1}^K e^{\theta_j^\top x_i}}
$$

Since in Deep Learning tasks the number of points in a dataset could be really huge, we usually use {%include link.html title='Stochastic gradient descent'%} based approaches as a workhorse. 

In such algorithms one uses the estimation of a gradient at each step instead of the full gradient vector, for example, in cross entropy we have:

$$
\nabla_\theta L(\theta) = \dfrac{1}{n} \sum\limits_{i=1}^n \left( h_\theta(x_i) - y_i \right) x_i^\top
$$

The simplest approximation is statistically judged unbiased estimation of a gradient:

$$
g(\theta) = \dfrac{1}{b} \sum\limits_{i=1}^b \left( h_\theta(x_i) - y_i \right) x_i^\top\approx \nabla_\theta L(\theta)
$$

where we initially sample randomly only $$b \ll n$$ points and calculate sample average. It can be also considered as a noisy version of the full gradient approach.

![](../MLP_optims.svg)


# Code
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Deep%20learning.ipynb)

# References
* [Optimization for Deep Learning Highlights in 2017](http://ruder.io/deep-learning-optimization-2017/)
* [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/)