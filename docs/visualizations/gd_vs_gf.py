"""Gradient descent (explicit Euler) vs gradient flow (continuous limit).

Same ill-conditioned quadratic, same starting point. GD with the optimal
worst-case step zig-zags across the valley; the gradient flow is the smooth
ODE trajectory it discretizes.
"""
from __future__ import annotations

from flows_common import OPTIM_VIS, render_trajectory_poster, render_trajectory_video, spiral_problem

GD_BLUE = "#2E6FE0"
GF_RED = "#FF3B30"


def specs(prob):
    return [
        {"points": prob["gf"], "color": GF_RED, "kind": "curve", "markers": False, "label": "Gradient Flow"},
        {"points": prob["gd"], "color": GD_BLUE, "kind": "zigzag", "markers": True, "label": "Gradient Descent"},
    ]


CORNERS = [
    ("Gradient Descent", GD_BLUE, (0.66, 0.97)),
    ("Gradient Flow", GF_RED, (0.66, 0.88)),
]


def main():
    prob = spiral_problem()
    s = specs(prob)
    render_trajectory_poster(OPTIM_VIS / "gd_vs_gf_poster.pdf", prob, s, corner_labels=CORNERS)
    render_trajectory_video(OPTIM_VIS / "gd_vs_gf.mp4", prob, s, corner_labels=CORNERS, frames=240, fps=60)


if __name__ == "__main__":
    main()
