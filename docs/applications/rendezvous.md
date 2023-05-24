---
layout: default
title: Rendezvous problem
parent: Applications
---

# Problem

![](../rendezvous.svg)

We have two bodies in discrete time: the first is described by its coordinate $$x_i$$ and its speed $$v_i$$, the second has coordinate $$z_i$$ and speed $$u_i$$. Each body has its own dynamics, which we denote as linear systems with matrices $$A, B, C, D$$:

$$
\begin{align*}
x_{i+1} = Ax_i + Bu_i \\
z_{i+1} = Cz_i + Dv_i
\end{align*}
$$

We want these bodies to meet in future at some point $$T$$ in such a way, that preserve minimum energy through the path. We will consider only kinetic energy, which is proportional to the squared speed at each point of time, that's why optimization problem takes the following form:

$$
\begin{align*}
& \min \sum_{i=1}^T \|u_i\|_2^2 + \|v_i\|_2^2 \\
\text{s.t. } & x_{t+1} = Ax_t + Bu_t, \; t = 1,\ldots,T-1\\
& z_{t+1} = Cz_t + Dv_t, \; t = 1,\ldots,T-1\\
& x_T = z_T
\end{align*}
$$

Problem of this type arise in space engineering - just imagine, that the first body is the spaceship, while the second, say, Mars.

# Code
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Rendezvous.ipynb)

# References
* [Jupyter notebook](https://colab.research.google.com/github/amkatrutsa/MIPT-Opt/blob/master/01-Intro/demos.ipynb#scrollTo=W264L1t1p3mF) by A. Katrutsa
