---
title: "Lyapunov energy of the accelerated flow (strongly convex)"
---

For $\mu$-strongly convex $f$ the right flow has **constant** damping
$\ddot X + 2\sqrt{\mu}\,\dot X + \nabla f(X)=0$ and converges **linearly**. The
Lyapunov energy
$$
E(t)=\bigl(f(X(t))-f^*\bigr)+\tfrac12\bigl\|\dot X(t)+\sqrt{\mu}\,(X(t)-x^*)\bigr\|^2
$$
satisfies $\dot E\le-\sqrt{\mu}\,E$, hence $E(t)\le E(0)e^{-\sqrt{\mu}\,t}$ and
$f(X(t))-f^*\le E(0)e^{-\sqrt{\mu}\,t}$ — a straight line on the semi-log plot.

:::{.video}
lyapunov_energy_sc.mp4
:::

[Code](lyapunov_energy_sc.py)
