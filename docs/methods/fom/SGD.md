---
layout: default
title: Stochastic Gradient Descent
parent: First Order Methods
grand_parent: Methods
nav_order: 2
---

## Summary
A classical problem of minimizing finite sum of the smooth and convex functions was considered. 
$$
\underset{x \in \mathbb{R}^{p}}{\operatorname{minimize}} g(x) :=\frac{1}{n} \sum{i=1}^{n} f{i}(x)
$$
This problem usually arise in Deep Learning, where the gradient of the loss function is calculating over the huge number of data points, which could be very expensive in terms of the iteration cost. Baseline solution to the problem is to calculate the loss function and the corresponding gradient vector only on the small subset of indicies from  , which usually referrs as Stochastic Gradient Descent(SGD)
##Contributions
##Opportunities