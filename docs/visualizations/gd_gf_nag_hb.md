---
title: "GD / GF / NAG-SC / Polyak HB"
---

On a $\mu$-strongly convex quadratic $f(x)=\tfrac12 x^\top A x$ four dynamics
race to the minimizer. Gradient descent and the gradient flow descend
monotonically; the accelerated schemes carry **momentum** and overshoot:
$$
\text{NAG-SC:}\;\; \ddot x + 2\sqrt{\mu}\,\dot x + \nabla f(x)=0,
\qquad
\text{HB:}\;\; x_{k+1}=x_k-\alpha\nabla f(x_k)+\beta(x_k-x_{k-1}),
$$
with Polyak's optimal $\alpha^\star=\tfrac{4}{(\sqrt L+\sqrt\mu)^2}$,
$\beta^\star=\bigl(\tfrac{\sqrt L-\sqrt\mu}{\sqrt L+\sqrt\mu}\bigr)^2$. The
oscillations are the price (and the engine) of acceleration.

:::{.video}
gd_gf_nag_hb.mp4
:::

[Code](gd_gf_nag_hb.py)
