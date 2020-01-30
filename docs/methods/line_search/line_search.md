---
layout: default
title: Line search
parent: Methods
has_children: True
nav_order: 0
---
# Problem

Suppose, we have a problem of minimization function $f(x): \mathbb{R} \to \mathbb{R}$ of scalar variable:
$$
f(x) \to \min_{x \in \mathbb{R}}
$$

Sometimes, we refer to the similar problem of finding minimum on the line segment $[a,b]$:

$$
f(x) \to \min_{x \in [a,b]}
$$

Line search is very important building block in many optimization algotithms.

# Code
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Line_search.ipynb)