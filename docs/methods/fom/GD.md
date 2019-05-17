---
layout: default
title: Gradient Descent
parent: First Order Methods
grand_parent: Methods
nav_order: 1
---

{% include tabs.html bibtex = '@article{cauchy1847methode,
  title={M{\'e}thode g{\'e}n{\'e}rale pour la r{\'e}solution des systemes d’{\'e}quations simultan{\'e}es}, author={Cauchy, Augustin},
  journal={Comp. Rend. Sci. Paris},
  volume={25},
  number={1847},
  pages={536--538},
  year={1847}
}' file='/assets/files/GD.pdf'%}

## Summary
A classical problem of function minimization is considered. 

$$
\tag{GD}
x_{k+1} = x_k - \eta_k\nabla f(x_k)
$$

* The bottleneck (for almost all gradient methods) is choosing step-size, which can lead to the dramatic difference in method's behaviour. 
* One of the theoretical suggestions: choosing stepsize inversly proportional to the gradient Lipschitz constant $\eta_k = \dfrac{1}{L}$
* In huge-scale applications the cost of iteration is usually defined by the cost of gradient calculation (at least $\mathcal{O}(p)$)

## Bounds

| Conditions | $f(x_k) - f(x^*) \leq$ | Type of convergence | $\| x_k - x^* \| \leq$ |
| ---------- | ---------------------- | ------------------- | --------------------- |
| Convex<br/>Lipschitz-continious function($G$) | $\mathcal{O}\left(\dfrac{1}{k} \right) \; \dfrac{GR}{k}$ | Sublinear |                       |
| Convex<br/>Lipschitz-continious gradient ($L$) | $\mathcal{O}\left(\dfrac{1}{k} \right) \; \dfrac{LR^2}{k}$ | Sublinear |                       |
| $\mu$-Strongly convex<br/>Lipschitz-continious hessian($M$) |                        | Locally linear<br /> $R < \overline{R}$ | $\dfrac{\overline{R}R}{\overline{R} - R} \left( 1 - \dfrac{2\mu}{L+3\mu}\right)$ |

* $R = \| x_0 - x^*\|$ - initial distance
* $\overline{R} = \dfrac{2\mu}{M}$

## Materials

* [The zen of gradient descent. Moritz Hardt](http://blog.mrtz.org/2013/09/07/the-zen-of-gradient-descent.html)