"""Gradient descent (optimal step) vs proximal point with a larger step.

PPM is unconditionally stable, so it can take a much larger step than GD without
overshooting. Here GD uses its optimal worst-case step alpha*_GD = 2/(mu+L),
while PPM uses a step twice as large and reaches the minimizer in fewer, smoother
iterations.
"""
from __future__ import annotations

from flows_common import OPTIM_VIS, render_trajectory_poster, render_trajectory_video, spiral_problem

GD_BLUE = "#2E6FE0"
PPM_RED = "#FF3B30"

TITLE = "Gradient Descent vs Proximal Point (optimal step)"
DESCRIPTION = r"""
For the quadratic $f(x)=\tfrac12 x^\top A x$ every PPM eigen-factor
$1/(1+\alpha\lambda_i)$ is below one for **any** $\alpha>0$ — the method never
diverges. So while gradient descent is capped at $\alpha^\star_{GD}=2/(\mu+L)$,
the proximal point method can safely take a much larger step
$\alpha^\star_{PPM}$ and contracts faster, reaching the minimizer in fewer,
smoother iterations.
"""


def build():
    prob = spiral_problem()
    a_gd = prob["alpha_gd"]
    a_ppm = prob["alpha_ppm"]
    ppm = prob["ppm_iter"](a_ppm, n=6)
    specs = [
        {"points": ppm, "color": PPM_RED, "smooth": True, "markers": True, "label": "Proximal Point"},
        {"points": prob["gd"], "color": GD_BLUE, "smooth": False, "markers": True, "label": "Gradient Descent"},
    ]
    corners = [
        ("Gradient Descent", GD_BLUE, (0.58, 0.97)),
        (rf"$\alpha=\alpha^\star_{{GD}}\approx {a_gd:.2f}$", GD_BLUE, (0.58, 0.90)),
        ("Proximal Point (optimal step)", PPM_RED, (0.58, 0.80)),
        (rf"$\alpha=\alpha^\star_{{PPM}}\approx {a_ppm:.2f}$", PPM_RED, (0.58, 0.73)),
    ]
    return prob, specs, corners


def main():
    prob, specs, corners = build()
    render_trajectory_poster(OPTIM_VIS / "gd_vs_ppm_optimal_poster.pdf", prob, specs, corner_labels=corners)
    render_trajectory_video(OPTIM_VIS / "gd_vs_ppm_optimal.mp4", prob, specs, corner_labels=corners, frames=240, fps=60)


if __name__ == "__main__":
    main()
