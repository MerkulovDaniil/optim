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
A classical problem of minimizing finite sum of the smooth and convex functions was considered. 

$$
\tag{GD}
x_{k+1} = x_k - \eta_k\nabla f(x_k)
$$

