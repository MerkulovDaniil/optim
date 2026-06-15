---
title: "Escaping a local minimum: GF vs SDE"
---

On a double-well potential the deterministic **gradient flow**
$\dot x=-f'(x)$ is trapped in its starting basin. The **stochastic flow**
$$
dx(t) = -f'\bigl(x(t)\bigr)\,dt + \sigma\,dW(t)
$$
keeps exploring: it climbs over the barrier and its density relaxes to the
**Gibbs distribution** $\rho^*(x)\propto e^{-2f(x)/\sigma^2}$, concentrated on the
global minimum. The bottom bars track the fraction of trajectories that have
reached the global well.

:::{.video}
stochastic_escape.mp4
:::

[Code](stochastic_escape.py)
