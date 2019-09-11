---
layout: default
title: "Lookahead Optimizer: $$k$$ steps forward, $$1$$ step back"
parent: First order methods
grand_parent: Methods
nav_order: 4
---

{% include tabs.html bibtex = '@article{zhang2019lookahead,
  title={Lookahead Optimizer: k steps forward, 1 step back},
  author={Zhang, Michael R and Lucas, James and Hinton, Geoffrey and Ba, Jimmy},
  journal={arXiv preprint arXiv:1907.08610},
  year={2019}
}' file='https://arxiv.org/pdf/1907.08610'%}

# Summary

The lookahead method provides an interesting way to accelerate and stabilize algorithms of stochastic gradient descent family.
Main idea is quite simple: 

* Set some number $$k$$. Take initial parameter weights $$\theta_0 = \hat{\theta}_0$$
* Do $$k$$ steps with your favorite optimization algorithm: $$\hat{\theta}_1, \ldots, \hat{\theta}_k$$
* Take some value between initial $$\theta_0$$ and $$\hat{\theta}_k$$:

    $$
    \theta_{t+1} = (1 - \alpha)\theta_{t} + \alpha\hat{\theta_k}
    $$

* Update $$\hat{\theta_0}$$ with the last output of the algorithm.
* Repeat ~~profit~~

Authors introduced separation on the *fast weights* and *slow weights*, which naturally arises in the described procedure.
The paper contains proof for optimal step-size of the quadratic loss function and provides understanding why this technique could reduce variance of {% include link.html title="Stochastic gradient descent" %} in the noisy quadratic case. Moreover, this work compares the convergence rate in dependency of condition number of the squared system.

It is worth to say, that autor claims significant improvement in practical huge scale settings (ImageNet, CIFAR10,CIFAR100)
