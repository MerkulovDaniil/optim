---
layout: default
parent: Convex sets
grand_parent: Theory
title: Affine set
nav_order: 1
---

# Line
Suppose $$x_1, x_2 $$ are two points in $$\mathbb{R^n}$$. Then the line passing through them is defined as follows:

$$
x = \theta x_1 + (1 - \theta)x_2, \theta \in \mathbb{R}
$$

![](../line.svg)

# Affine set
The set $$A$$ is called **affine** if for any $$x_1, x_2$$ from $$A$$ the line passing through them also lies in $$A$$, i.e. $$\forall \theta \in \mathbb{R}, \forall x_1, x_2 \in A: \theta x_1 + (1- \theta) x_2 \in

## Examples: 
* \$$\mathbb{R}^n$$
* The set of solutions $$ \left\{ x \mid \mathbf{A}x = \mathbf{b}\right\} $$

# Related definitions
## Affine combination
Let we have $$x_1, x_2, \ldots, x_k \in S$$, then the point $$\theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_k x_k$$ is called affine combination of $$x_1, x_2, \ldots, x_k$$ if $$\sum\limits_{i=1}^k\theta_i = 1$$

## Affine hull
The set of all affine combinations of points in set $$S$$ is called the affine hull of $$S$$:

$$
\mathbf{aff}(S) = \left\{ \sum\limits_{i=1}^k\theta_i x_i \mid x_i \in S, \sum\limits_{i=1}^k\theta_i = 1\right\}
$$