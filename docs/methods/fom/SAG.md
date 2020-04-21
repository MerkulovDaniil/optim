---
layout: default
title: Stochastic average gradient
parent: First order methods
grand_parent: Methods
nav_order: 6
bibtex: |
  @article{schmidt2017minimizing,
  title={Minimizing finite sums with the stochastic average gradient},
  author={Schmidt, Mark and Le Roux, Nicolas and Bach, Francis},
  journal={Mathematical Programming},
  volume={162},
  number={1-2},
  pages={83--112},
  year={2017},
  publisher={Springer}
  }
file: assets/files/SAG.pdf
---
## Summary
A classical problem of minimizing finite sum of the smooth and convex functions was considered. 

$$
\min\limits_{x \in \mathbb{R}^{p}} g(x) := \frac{1}{n} \sum_{i=1}^{n} f_i(x)
$$


This problem usually arises in Deep Learning, where the gradient of the loss function is calculating over the huge number of data points, which could be very expensive in terms of the iteration cost. Baseline solution to the problem is to calculate the loss function and the corresponding gradient vector only on the small subset of indicies from $$i = 1, \ldots, n$$, which usually refers as {% include link.html title='Stochastic gradient descent'%}. The authors claims, that the convergence rate of proposed algorithm is the same a for the full {% include link.html title='Gradient descent'%} method ($$\mathcal{O}\left( \dfrac{1}{k}\right)$$ for convex functions and $$\mathcal{O}\left( \dfrac{1}{\sqrt{k}}\right)$$ for strongly convex objectives), but the iteration costs remains the same as for the stochastic version.

The method itself takes the following form:

$$
\tag{SAG} 
x_{k+1}=x_{k}-\frac{\alpha_{k}}{p} \sum_{i=1}^{p} y^{i}_{k}
$$

where at each iteration only random summand of a gradient is updated:

$$
\tag{SAG} 
y^{i}_{k}=\left\{\begin{array}{ll}{f_{i}^{\prime}\left(x_{k}\right)} & {\text { if } i=i_{k}} \\ {y^{i}_{k-1}} & {\text { otherwise }}\end{array}\right.
$$

* There is a dependency on dimensionality factor $n$ in bounds. However, it can be improved using restart technique.
* Empirical results were only shown on logistic regression with Tikhonov regularization problems on different datasets.
* Batch and non- uniform versions are also presented in the paper.
* The first known paper, that contains proof of linear convergence for the convex case.

## Bounds
For a constant step size $\alpha = \dfrac{1}{16 L}$, where $L$ stands for the Lipschitz constant of a gradient of each function $ f_i(x) $ (in practice, it means that $ L = \max\limits_{i=1, \ldots, n} L_i $).

$$
\mathbb{E}\left[g\left(\overline{x}_{k}\right)\right]-g\left(x^{*}\right) \leqslant \frac{32 n}{k} C_{0},
$$

where $ C_0=g\left(x_0\right)-g\left(x^*\right)+\frac{4L}{n} \\| x_0 - x^\ast\\|^2 +\frac{\sigma^2}{16L}$  in convex case and

$$
\mathbb{E}\left[g\left(x_{k}\right)\right]-g\left(x^*\right) \leqslant\left(1-\min \left\{\frac{\mu}{16 L}, \frac{1}{8 n}\right\}\right)^{k} C_{0}
$$

in $\mu$ - strongly convex case.



