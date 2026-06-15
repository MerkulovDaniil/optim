---
title: "Discretizing the gradient flow: GD, PPM, NAG, Heavy Ball"
---

Every first-order method is a discretization of a continuous-time flow. Below,
four scenes on the same kind of quadratic landscape contrast the discrete
iterates with the smooth ODE trajectory they approximate.

## Gradient descent vs gradient flow

Gradient descent is the **explicit Euler** discretization of the gradient flow:
$$
\frac{x_{k+1}-x_k}{\alpha} = -\nabla f(x_k)
\quad\xrightarrow{\;\alpha\to 0\;}\quad
\dot x(t) = -\nabla f\bigl(x(t)\bigr).
$$
With the optimal step $\alpha^\star=2/(\mu+L)$, GD **zig-zags** across the steep
valley while the gradient flow follows the smooth trajectory GD tries to track.

:::{.video}
gd_vs_gf.mp4
:::

[Code](gd_vs_gf.py)

## Gradient descent vs proximal point (same step)

The proximal point method is the **implicit Euler** discretization
$x_{k+1}=\operatorname{prox}_{\alpha f}(x_k)=(I+\alpha A)^{-1}x_k$. At the *same*
step where GD already zig-zags, every PPM eigen-factor $1/(1+\alpha\lambda_i)$
stays in $(0,1)$, so the proximal iterates approach the minimizer smoothly.

:::{.video}
gd_vs_ppm.mp4
:::

[Code](gd_vs_ppm.py)

## Proximal point with a larger step

PPM is unconditionally stable for any $\alpha>0$, so it can safely take a step
far larger than GD's $\alpha^\star_{GD}=2/(\mu+L)$ and contract faster, reaching
the minimizer in fewer, smoother iterations.

:::{.video}
gd_vs_ppm_optimal.mp4
:::

[Code](gd_vs_ppm_optimal.py)

## GD / GF / NAG-SC / Polyak Heavy Ball

On a strongly convex quadratic, the accelerated schemes carry **momentum** and
overshoot:
$$
\text{NAG-SC:}\;\; \ddot x + 2\sqrt{\mu}\,\dot x + \nabla f(x)=0,
\qquad
\text{HB:}\;\; x_{k+1}=x_k-\alpha\nabla f(x_k)+\beta(x_k-x_{k-1}).
$$
The oscillations are the price — and the engine — of acceleration.

:::{.video}
gd_gf_nag_hb.mp4
:::

[Code](gd_gf_nag_hb.py)
