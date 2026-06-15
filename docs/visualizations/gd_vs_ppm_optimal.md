---
title: "Gradient Descent vs Proximal Point (optimal step)"
---

For the quadratic $f(x)=\tfrac12 x^\top A x$ every PPM eigen-factor
$1/(1+\alpha\lambda_i)$ is below one for **any** $\alpha>0$ — the method never
diverges. So while gradient descent is capped at $\alpha^\star_{GD}=2/(\mu+L)$,
the proximal point method can safely take a much larger step
$\alpha^\star_{PPM}$ and contracts faster, reaching the minimizer in fewer,
smoother iterations.

:::{.video}
gd_vs_ppm_optimal.mp4
:::

[Code](gd_vs_ppm_optimal.py)
