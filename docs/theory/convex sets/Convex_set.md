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
x = \theta x_1 + (1 - \theta)x_2, \; \theta \in [0,1]
$$

![center](../line_segment.svg)

# Convex set
The set $$S$$ is called **convex** if for any $$x_1, x_2$$ from $$S$$ the line segment between them also lies in $$S$$, i.e. 

$$
\forall \theta \in [0,1], \; \forall x_1, x_2 \in S: \\ \theta x_1 + (1- \theta) x_2 \in S
$$

## Examples: 

* Any affine set
* Ray
* Line segment

![center](../convex_1.svg)

![center](../convex_2.svg)

# Related definitions
## Convex combination
Let $$x_1, x_2, \ldots, x_k \in S$$, then the point $$\theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_k x_k$$ is called the convex combination of points $$x_1, x_2, \ldots, x_k$$ if $$\sum\limits_{i=1}^k\theta_i = 1, \; \theta_i \ge 0$$

## Convex hull
The set of all convex combinations of points from $$S$$ is called the convex hull of the set $$S$$.

$$
\mathbf{conv}(S) = \left\{ \sum\limits_{i=1}^k\theta_i x_i \mid x_i \in S, \sum\limits_{i=1}^k\theta_i = 1, \; \theta_i \ge 0\right\}
$$

* The set $$\mathbf{conv}(S)$$ is the smallest convex set containing $$S$$.
* The set $$S$$ is convex if and only if $$S = \mathbf{conv}(S)$$.


Examples:
![center](../convex_hull.svg)

# Finding convexity

In practice it is very important to understand whether a specific set is convex or not. Two approaches are used for this depending on the context.
* By definition.
* Show that $$S$$ is derived from simple convex sets using operations that preserve convexity.

## By definition

$$
x_1, x_2 \in S, \; 0 \le \theta \le 1 \;\; \rightarrow \;\; \theta x_1 + (1-\theta)x_2 \in S
$$

## Preserving convexity

### The linear combination of convex sets is convex

Let there be 2 convex sets $$S_x, S_y$$, let the set $$S = \left\{s \mid s = c_1 x + c_2 y, \; x \in S_x, \; y \in S_y, \; c_1, c_2 \in \mathbb{R}\right\}$$

Take two points from $$S$$: $$s_1 = c_1 x_1 + c_2 y_1, s_2 = c_1 x_2 + c_2 y_2$$ and prove that the segment between them $$ \theta  s_1 + (1 - \theta)s_2, \theta \in [0,1] $$ also belongs to $$S$$

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

### The intersection of any (!) number of convex sets is convex

If the desired intersection is empty or contains one point, the property is proved by definition. Otherwise, take 2 points and a segment between them. These points must lie in all intersecting sets, and since they are all convex, the segment between them lies in all sets and, therefore, in their intersection.

### The image of the convex set under affine mapping is convex

$$
S \subseteq \mathbb{R}^n \text{ convex}\;\; \rightarrow \;\; f(S) = \left\{ f(x) \mid x \in S \right\} \text{ convex} \;\;\;\; \left(f(x) = \mathbf{A}x + \mathbf{b}\right)
$$

Examples of affine functions: extension, projection, transposition, set of solutions of linear matrix inequality $$\left\{ x \mid x_1 A_1 + \ldots + x_m A_m \preceq B\right\}$$ Here $$A_i, B \in \mathbf{S}^p$$ are symmetric matrices $$p \times p$$. 

Note also that the prototype of the convex set under affine mapping is also convex.

$$
S \subseteq \mathbb{R}^m \text{ convex}\; \rightarrow \; f^{-1}(S) = \left\{ x \in \mathbb{R}^n \mid f(x) \in S \right\} \text{ convex} \;\; \left(f(x) = \mathbf{A}x + \mathbf{b}\right)
$$
