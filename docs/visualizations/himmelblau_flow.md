---
title: "Gradient flow on Himmelblau"
---

Himmelblau's function
$f(x,y)=(x^2+y-11)^2+(x+y^2-7)^2$
has **four** equal global minima. The gradient flow $\dot x=-\nabla f(x)$ sends
each starting point to one of them; coloring trajectories by their limit reveals
the four **basins of attraction**. On a non-convex landscape the initial point
alone decides the outcome.

:::{.video}
himmelblau_flow.mp4
:::

[Code](himmelblau_flow.py)
