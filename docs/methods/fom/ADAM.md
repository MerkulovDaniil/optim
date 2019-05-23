---
layout: default
title: ADAM: A Method for Stochastic Optimization
parent: First Order Methods
grand_parent: Methods
nav_order: 4
---

{% include tabs.html bibtex = '@article{kingma2014adam,
  title={Adam: A method for stochastic optimization},
  author={Kingma, Diederik P and Ba, Jimmy},
  journal={arXiv preprint arXiv:1412.6980},
  year={2014}
}' file='assets/files/ADAM.pdf'%}

## Summary

Adam is the stochastic first order optimization algorithm, that uses historical information about stochastic gradients and incorporates it in attempt to estimate second order moment of stochastic gradients.

$$
\tag{ADAM}
\begin{align*}
x_{k+1} &= x_k - \alpha_k \dfrac{\widehat{m_k}}{\sqrt{\widehat{v_k}} + \epsilon} \\
\tag{First moment estimation}
\widehat{m_k} &= \dfrac{m_k}{1 - \beta_1^k} \\
m_k &= \beta_1 m_{k-1} + (1 - \beta_1) g_k \\
\tag{Second moment estimation}
\widehat{v_k} &= \dfrac{v_k}{1 - \beta_2^k} \\
v_k &= \beta_2 v_{k-1} + (1 - \beta_2)g_k^2 \\
\end{align*}
$$

All vector operations are element-wise. $\alpha = 0.001, \beta_1 = 0.9, \beta_2 = 0.999$ - the default values for hyperparameters ($\epsilon$ here is needed for avoiding zero division problems) and $g_k = \nabla f(x_k, \xi_k)$ is the sample of stochastic gradient.

* We can consider this approach as normalization of each parameter buy using individual learning rates on $ \mathcal{N} (0,1)$, since $\mathbb{E}_{\xi_k}[g_k] = \mathbb{E}_{\xi_k}[\widehat{m_k}]$ and $\mathbb{E}_{\xi_k}[g_k \odot g_k] = \mathbb{E}_{\xi_k}[\widehat{v_k}]$
* There are some [issues](https://www.fast.ai/2018/07/02/adam-weight-decay/) with Adam effectiveness and some [works](https://arxiv.org/pdf/1705.08292.pdf), stated, that adaptive metrics methods could lead to worse generalization.
* The name came from "**Ada**ptive **M**oment estimation"

## Bounds

| Conditions | $\Vert \mathbb{E} [f(x_k)] - f(x^*)\Vert \leq$ | Type of convergence | $\Vert \mathbb{E}[x_k] - x^* \Vert \leq$ |
| ---------- | ---------------------- | ------------------- | --------------------- |
| Convex | $ \mathcal{O}\left(\dfrac{1}{\sqrt{k}} \right) $ | Sublinear |                       |

Version of Adam for a strongly convex functions is considered i [this](https://arxiv.org/pdf/1905.02957.pdf) work. The obtained rate is $ \mathcal{O}\left(\dfrac{\log k}{\sqrt{k}} \right) $, while the version for truly linear rate remains undiscovered.