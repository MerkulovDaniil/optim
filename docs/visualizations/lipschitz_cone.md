---
title: "Lipschitz continuity of a function"
---

A function $f: \mathbb{R}^n \to \mathbb{R}$ is called $G$-Lipschitz continuous on a set $S$ if there exists a constant $G \geq 0$ such that for all $x, y \in S$:

$$
|f(x) - f(y)| \leq G \|x - y\|
$$

Geometrically, the Lipschitz condition means the function graph lies inside a cone of slope $G$ centered at any point $(x, f(x))$. The animation below illustrates how moving the reference point shifts the cone while the function always remains inside it.

![Lipschitz continuity](lipschitz_cone.svg)

:::{.video}
lipschitz_cone.mp4
:::
