---
layout: default
title: Stochastic gradient descent
parent: First order methods
grand_parent: Methods
nav_order: 2
bibtex: |
  @article{robbins1951stochastic,
  title={A stochastic approximation method},
  author={Robbins, Herbert and Monro, Sutton},
  journal={The annals of mathematical statistics},
  pages={400--407},
  year={1951},
  publisher={JSTOR}
  }
file: /assets/files/SGD.pdf
---
## Summary
Suppose, our target function is the sum of functions.

$$
\min\limits_{x \in \mathbb{R}^{p}} g(x) := \frac{1}{n} \sum_{i=1}^{n} f_i(x)
$$

This problem usually arises in Deep Learning, where the gradient of the loss function is calculating over the huge number of data points, which could be very expensive in terms of the iteration cost. 

## Bounds

| Conditions | $\Vert \mathbb{E} [f(x_k)] - f(x^*)\Vert \leq$ | Type of convergence | $\Vert \mathbb{E}[x_k] - x^* \Vert \leq$ |
| ---------- | ---------------------- | ------------------- | --------------------- |
| Convex | $ \mathcal{O}\left(\dfrac{1}{\sqrt{k}} \right) $ | Sublinear |                       |
| $ \mu$-Strongly convex | $\mathcal{O}\left(\dfrac{1}{k} \right) $ | Sublinear |                       |



## Materials
