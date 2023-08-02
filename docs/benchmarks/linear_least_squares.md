---
layout: default
title: Linear Least Squares
parent: Benchmarks
---

# Problem 

In a least-squares, or linear regression, problem, we have measurements $$ A \in \mathbb{R}^{m \times n} $$ and $$ b \in \mathbb{R}^{m} $$ and seek a vector $$ x \in \mathbb{R}^{n} $$ such that $$ A x $$ is close to $$ b $$. Closeness is defined as the sum of the squared differences: 

$$
f(x) = \|Ax - b\|_2^2 \to \min_{x \in \mathbb{R^n}}
$$

