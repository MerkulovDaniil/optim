---
layout: default
parent: Convex sets
grand_parent: Theory
title: Affine set
nav_order: 1
---

# Line
Даны 2 точки $$x_1, x_2 \in \mathbb{R^n}$$. Тогда прямая, проходящая через них определяется следующим образом:

$$
x = \theta x_1 + (1 - \theta)x_2, \theta \in \mathbb{R}
$$

![](../line.svg)

# Affine set
Множество $$A$$ называется афинным, если для любых $$x_1, x_2$$ из $$A$$ прямая, проходящая через них так же лежит в $$A$$, т.е. $$\forall \theta \in \mathbb{R}, \forall x_1, x_2 \in A: \theta x_1 + (1- \theta) x_2 \in A$$

## Examples: 
* \$$\mathbb{R}^n$$
* Множество $$ \left\{ x \mid \mathbf{A}x = \mathbf{b}\right\} $$

# Related definitions
## Affine combination
Пусть $$x_1, x_2, \ldots, x_k \in S$$, тогда точка $$\theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_k x_k$$ называется афинной комбинацией точек $$x_1, x_2, \ldots, x_k$$ при условии $$\sum\limits_{i=1}^k\theta_i = 1$$

## Affine hull
Множество всех афинных комбинаций точек множества $$S$$ называется афинной оболочкой множества $$S$$.

$$
\mathbf{aff}(S) = \left\{ \sum\limits_{i=1}^k\theta_i x_i \mid x_i \in S, \sum\limits_{i=1}^k\theta_i = 1\right\}
$$