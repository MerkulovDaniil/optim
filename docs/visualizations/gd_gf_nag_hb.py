"""Four methods on one strongly convex quadratic: GD, GF, NAG-SC, Polyak HB.

GD and GF crawl monotonically; the accelerated schemes (Nesterov for the
strongly convex case and Polyak's heavy ball with optimal parameters) overshoot
and oscillate around the minimizer while converging much faster.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from flows_common import OPTIM_VIS, render_trajectory_poster, render_trajectory_video, resample_by_arclength, write_page

GD_BLUE = "#2E6FE0"
GF_RED = "#FF3B30"
NAG_GREEN = "#00A36C"
HB_MAGENTA = "#C026D3"

TITLE = "GD / GF / NAG-SC / Polyak HB"
DESCRIPTION = r"""
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
"""


def build():
    theta = np.deg2rad(-24.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mu, L = 1.0, 48.0
    A = rot @ np.diag([mu, L]) @ rot.T
    x0 = np.array([-2.4, 1.7])

    def grad(x):
        return A @ x

    # gradient flow
    sol = solve_ivp(lambda _t, x: -grad(x), (0.0, 14.0), x0, t_eval=np.linspace(0.0, 14.0, 1500), rtol=1e-9, atol=1e-11)
    gf = resample_by_arclength(sol.y.T, 600)

    # gradient descent (slightly aggressive step so the zig-zag is visible)
    a_gd = 1.7 / L
    x = x0.copy()
    gd = [x.copy()]
    for _ in range(80):
        x = x - a_gd * grad(x)
        gd.append(x.copy())
    gd = np.asarray(gd)

    # NAG, strongly convex constant momentum
    beta_nag = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
    xk, xkm = x0.copy(), x0.copy()
    nag = [xk.copy()]
    for _ in range(90):
        y = xk + beta_nag * (xk - xkm)
        xn = y - (1.0 / L) * grad(y)
        xkm, xk = xk, xn
        nag.append(xk.copy())
    nag = np.asarray(nag)

    # Polyak heavy ball, optimal parameters
    a_hb = 4.0 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    beta_hb = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
    xk, xkm = x0.copy(), x0.copy()
    hb = [xk.copy()]
    for _ in range(90):
        xn = xk - a_hb * grad(xk) + beta_hb * (xk - xkm)
        xkm, xk = xk, xn
        hb.append(xk.copy())
    hb = np.asarray(hb)

    allpts = np.vstack([gd, nag, hb, gf, x0[None, :]])
    pad = 0.5
    xlim = (allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
    ylim = (allpts[:, 1].min() - pad, allpts[:, 1].max() + pad)
    xg = np.linspace(*xlim, 320)
    yg = np.linspace(*ylim, 320)
    xx, yy = np.meshgrid(xg, yg)
    zz = 0.5 * (A[0, 0] * xx**2 + 2 * A[0, 1] * xx * yy + A[1, 1] * yy**2)

    prob = {"x0": x0, "xlim": xlim, "ylim": ylim, "grid": (xx, yy, zz), "minimizer": (0.0, 0.0)}
    # back-to-front so every curve stays readable: HB (widest swings) at the back,
    # the smooth GF on top.
    specs = [
        {"points": hb, "color": HB_MAGENTA, "smooth": False, "markers": False, "label": "Polyak HB"},
        {"points": gd, "color": GD_BLUE, "smooth": False, "markers": False, "label": "GD"},
        {"points": nag, "color": NAG_GREEN, "smooth": True, "markers": False, "label": "NAG SC"},
        {"points": gf, "color": GF_RED, "smooth": False, "markers": False, "label": "GF"},
    ]
    corners = [
        ("GD", GD_BLUE, (0.88, 0.97)),
        ("GF", GF_RED, (0.88, 0.90)),
        ("NAG SC", NAG_GREEN, (0.80, 0.83)),
        ("Polyak HB", HB_MAGENTA, (0.74, 0.76)),
    ]
    return prob, specs, corners


def main():
    prob, specs, corners = build()
    render_trajectory_poster(OPTIM_VIS / "gd_gf_nag_hb_poster.pdf", prob, specs, corner_labels=corners)
    render_trajectory_video(OPTIM_VIS / "gd_gf_nag_hb.mp4", prob, specs, corner_labels=corners, frames=260, fps=60)
    write_page("gd_gf_nag_hb", TITLE, DESCRIPTION, __file__)


if __name__ == "__main__":
    main()
