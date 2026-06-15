"""Gradient descent vs proximal point method at the *same* step size.

PPM is the implicit Euler discretization of the gradient flow:
x_{k+1} = (I + alpha A)^{-1} x_k. Unlike GD, every eigen-factor stays positive
for any alpha, so at the step where GD already zig-zags PPM moves smoothly and
monotonically toward the minimizer.
"""
from __future__ import annotations

from flows_common import OPTIM_VIS, render_trajectory_poster, render_trajectory_video, spiral_problem

GD_BLUE = "#2E6FE0"
PPM_RED = "#FF3B30"

TITLE = "Gradient Descent vs Proximal Point"
DESCRIPTION = r"""
The proximal point method is the **implicit Euler** discretization of the gradient flow:
$$
\frac{x_{k+1}-x_k}{\alpha}=-\nabla f(x_{k+1})
\;\Longleftrightarrow\;
x_{k+1}=\operatorname{prox}_{\alpha f}(x_k)=(I+\alpha A)^{-1}x_k .
$$
At the **same** step $\alpha=\alpha^\star_{GD}=2/(\mu+L)$ where gradient descent
already zig-zags, every eigen-factor $1/(1+\alpha\lambda_i)$ of PPM stays in
$(0,1)$, so the proximal iterates approach the minimizer smoothly and monotonically.
"""


def build():
    prob = spiral_problem()
    a = prob["alpha_gd"]
    ppm = prob["ppm_iter"](a, n=7)
    specs = [
        {"points": ppm, "color": PPM_RED, "smooth": True, "markers": True, "label": "Proximal Point"},
        {"points": prob["gd"], "color": GD_BLUE, "smooth": False, "markers": True, "label": "Gradient Descent"},
    ]
    corners = [
        ("Gradient Descent", GD_BLUE, (0.64, 0.97)),
        (rf"$\alpha=\alpha^\star\approx {a:.2f}$", GD_BLUE, (0.64, 0.90)),
        ("Proximal Point", PPM_RED, (0.64, 0.80)),
        (rf"$\alpha\approx {a:.2f}$", PPM_RED, (0.64, 0.73)),
    ]
    return prob, specs, corners


def main():
    prob, specs, corners = build()
    render_trajectory_poster(OPTIM_VIS / "gd_vs_ppm_poster.pdf", prob, specs, corner_labels=corners)
    render_trajectory_video(OPTIM_VIS / "gd_vs_ppm.mp4", prob, specs, corner_labels=corners, frames=240, fps=60)


if __name__ == "__main__":
    main()
