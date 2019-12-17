---
layout: default
parent: Convex sets
grand_parent: Theory
title: Convex set
nav_order: 2
---

# Line segment
Suppose $$x_1, x_2 $$ are two points in $$\mathbb{R^n}$$. Then the line segment between them is defined as follows:

$$
x = \theta x_1 + (1 - \theta)x_2, \theta \in [0,1]
$$

![center](../line_segment.svg)

# Convex set
The set $$S$$ is called **convex** if for any $$x_1, x_2$$ from $$S$$ the line segment between them also lies in $$S$$, i.e. 

$$
\forall \theta \in [0,1], \forall x_1, x_2 \in S: \\ \theta x_1 + (1- \theta) x_2 \in S
$$

## Examples: 

* Any affine set
* Ray
* Line segment

![center](../convex_1.svg)

![center](../convex_2.svg)

# Related definitions
## Convex combination
Пусть $$x_1, x_2, \ldots, x_k \in S$$, тогда точка $$\theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_k x_k$$ называется выпуклой комбинацией точек $$x_1, x_2, \ldots, x_k$$ при условии $$\sum\limits_{i=1}^k\theta_i = 1, \theta_i \ge 0$$

## Convex hull
Множество всех выпуклых комбинаций точек множества $$S$$ называется выпуклой оболочкой множества $$S$$.

$$
\mathbf{conv}(S) = \left\{ \sum\limits_{i=1}^k\theta_i x_i \mid x_i \in S, \sum\limits_{i=1}^k\theta_i = 1, \theta_i \ge 0\right\}
$$

Примеры:
![center](../convex_hull.svg)

# Finding convexity

На практике очень важно бывает понять, выпукло конкретное множество или нет. Для этого применяют 2 подхода в зависимости от контекста.
* By definition
* Показать, что $$S$$ получено из простых выпуклых множеств с помощью операций, сохраняющих выпуклость:

## By definition

$$
x_1, x_2 \in S,  0 \le \theta \le 1 \;\; \rightarrow \;\; \theta x_1 + (1-\theta)x_2 \in S
$$

## Показать, что $$S$$ получено из простых выпуклых множеств с помощью операций, сохраняющих выпуклость:

### Линейная комбинация выпуклых множеств выпукла

Пусть есть 2 выпуклых множества $$S_x, S_y$$, пусть множество $$S = \left\{s \mid s = c_1 x + c_2 y, x \in S_x, y \in S_y, c_1, c_2 \in \mathbb{R}\right\}$$

Возьмем две точки из $$S$$: $$s_1 = c_1 x_1 + c_2 y_1, s_2 = c_1 x_2 + c_2 y_2$$ и докажем, что отрезок между ними $$\theta s_1 + (1 - \theta)s_2, \theta \in [0,1]$$ так же принадлежит $$S$$

$$
\theta s_1 + (1 - \theta)s_2
$$

$$
\theta (c_1 x_1 + c_2 y_1) + (1 - \theta)(c_1 x_2 + c_2 y_2)
$$

$$
c_1 (\theta x_1 + (1 - \theta)x_2) + c_2 (\theta y_1 + (1 - \theta)y_2)
$$

$$
c_1 x + c_2 y \in S
$$

### Пересечение любого (!) числа выпуклых множеств выпукло


Если искомое пересечение пусто или содержит одну точку - свойство доказано по определению. В противном случае возьмем 2 точки и отрезок между ними. Эти точки должны лежать во всех пересекаемых множествах, а так как все они выпуклы, то и отрезок между ними лежит во всех множествах, а значит и в их пересечении.

###  Образ выпуклого множества при аффинном отображении выпуклый

$$
S \subseteq \mathbb{R}^n \text{ convex}\;\; \rightarrow \;\; f(S) = \left\{ f(x) \mid x \in S \right\} \text{ convex} \;\;\;\; \left(f(x) = \mathbf{A}x + \mathbf{b}\right)
$$

Примеры аффинных функций: растяжение, проекция, перенос, множество решений линейного матричного неравенства $$\left\{ x \mid x_1 A_1 + \ldots + x_m A_m \preceq B\right\}$$ Здесь $$A_i, B \in \mathbf{S}^p$$ - симметричные матрицы $$p \times p$$. 

Отметим так же, что прообраз выпуклого множества при аффинном отображении так же выпуклый.

$$
S \subseteq \mathbb{R}^m \text{ convex}\; \rightarrow \; f^{-1}(S) = \left\{ x \in \mathbb{R}^n \mid f(x) \in S \right\} \text{ convex} \;\; \left(f(x) = \mathbf{A}x + \mathbf{b}\right)
$$
