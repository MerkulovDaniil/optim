---
layout: default
parent: Convex sets
grand_parent: Theory
title: Conic set
nav_order: 3
---

# Convex cone
Множество $S$ называется выпуклым конусом, если:

$$
\forall x_1, x_2 \in S, \theta_1, \theta_2 \ge 0 \;\; \rightarrow \;\; \theta_1 x_1 + \theta_2 x_2 \in S
$$

![center](../convex_cone.svg)

#### Примеры:
$\mathbb{R}^n$; афинное множество, содержащее $0$; луч, $\mathbf{S}^n_+$ - множество симметричных положительно определенных матриц

### Неотрицательная коническая комбинация точек
Пусть $x_1, x_2, \ldots, x_k \in S$, тогда точка $\theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_k x_k$ называется неотрицательной конической комбинацией точек $x_1, x_2, \ldots, x_k$ при условии $\theta_i \ge 0$

### Коническая оболочка точек
Множество всех неотрицательных конических комбинаций точек множества $S$ называется конической оболочкой множества $S$.
$$\mathbf{cone}(S) = \left\{ \sum\limits_{i=1}^k\theta_i x_i \mid x_i \in S, \theta_i \ge 0\right\}$$

![center](../conic_hull.svg)