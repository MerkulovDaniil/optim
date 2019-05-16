---
layout: default
title: Stochastic Average Gradient
parent: First Order Methods
grand_parent: Methods
nav_order: 3
---

{% include tabs.html bibtex = '@article{schmidt2017minimizing,
  title={Minimizing finite sums with the stochastic average gradient},
  author={Schmidt, Mark and Le Roux, Nicolas and Bach, Francis},
  journal={Mathematical Programming},
  volume={162},
  number={1-2},
  pages={83--112},
  year={2017},
  publisher={Springer}
}'%}

## Summary
A classical problem of minimizing finite sum of the smooth and convex functions was considered. 

$$
\min\limits_{x \in \mathbb{R}^{p}} g(x) := \frac{1}{n} \sum_{i=1}^{n} f_i(x)
$$

This problem usually arise in Deep Learning, where the gradient of the loss function is calculating over the huge number of data points, which could be very expensive in terms of the iteration cost. Baseline solution to the problem is to calculate the loss function and the corresponding gradient vector only on the small subset of indicies from  , which usually referrs as Stochastic Gradient Descent(SGD) 

{% include link.html title='Stochastic Gradient Descent' %}

## Contributions

## Opportunities