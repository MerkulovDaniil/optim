---
layout: default
parent: Other
title: Neural ordinary differential equations
---

{% include tabs.html bibtex = '@inproceedings{chen2018neural,
  title={Neural ordinary differential equations},
  author={Chen, Tian Qi and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David K},
  booktitle={Advances in neural information processing systems},
  pages={6571--6583},
  year={2018}
}' file='https://arxiv.org/pdf/1806.07366.pdf'%}

# Summary

The main idea is the switching from discrete sequence of layers in neural network to continious hidden representation state, which is illustrated on the graph:

![](../odenet.svg)

It is common to consider such iterative transformation as an Euler discretization of the continous process. Conceptually, we can plot the following difference:

$$
\mathbf{h}_{t+1} = \mathbf{h}_t + f(\mathbf{h}_t, \theta_t) \; \to \; \dfrac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta)
$$

Authors highlighted the following benefits:
* Memory efficiency - there is a way to compute derivative of the scalar loss function without storing all the inner gradients.
* Adaptive computation - why the Euler's scheme if we can use modern ODE solvers?
* Parameter efficiency - parameters of the neighbours are adjusted together, which can be considered as a parameter reduction in supervised setting.
* Scalable and invertible normalizing flows - this model lead to the new class of invertible density models.
* Continious time-series models - this type of models allows to incorporate time series data at arbitrary measure points naturally (in contrast to usual discretization approach).

# Adjoint sensitivity method
ToDo - seems very interesting

# Related works
* [Additional dimensions in hidden state improves learning](https://arxiv.org/pdf/1904.01681.pdf)
* [Solving backward problem often causing instability](https://arxiv.org/pdf/1902.10298.pdf)