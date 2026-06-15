"""Gradient flow on Himmelblau's function: basins of attraction.

25 trajectories of x' = -grad f from a grid of starts. Himmelblau has four equal
global minima; each trajectory is colored by the basin it lands in. The gradient
grows like |x|^3, so we integrate densely and resample by arc length to keep the
curves visually smooth.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

from flows_common import OPTIM_VIS, add_watermark, make_writer, resample_by_arclength, write_page

MINIMA = np.array([[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]])
BASIN = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
LIM = 5.0

TITLE = "Gradient flow on Himmelblau"
DESCRIPTION = r"""
Himmelblau's function
$f(x,y)=(x^2+y-11)^2+(x+y^2-7)^2$
has **four** equal global minima. The gradient flow $\dot x=-\nabla f(x)$ sends
each starting point to one of them; coloring trajectories by their limit reveals
the four **basins of attraction**. On a non-convex landscape the initial point
alone decides the outcome.
"""


def himmel(w):
    x, y = w
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def grad(w):
    x, y = w
    a, b = x**2 + y - 11, x + y**2 - 7
    return np.array([4 * a * x + 2 * b, 2 * a + 4 * b * y])


def trajectories(n_vis=500):
    edge = 4.3
    starts = [(a, b) for a in np.linspace(-edge, edge, 5) for b in np.linspace(-edge, edge, 5)]
    trajs, colors = [], []
    for w0 in starts:
        sol = solve_ivp(lambda _t, w: -grad(w), (0, 8.0), list(w0), method="RK45", rtol=1e-9, atol=1e-11, dense_output=True, max_step=0.02)
        dense = sol.sol(np.linspace(0, 8.0, 8000)).T
        trajs.append(resample_by_arclength(dense, n_vis))
        colors.append(BASIN[int(np.argmin(np.linalg.norm(MINIMA - dense[-1], axis=1)))])
    return trajs, colors


def draw_landscape(ax):
    xs = np.linspace(-LIM, LIM, 400)
    xx, yy = np.meshgrid(xs, xs)
    zz = (xx**2 + yy - 11) ** 2 + (xx + yy**2 - 7) ** 2
    levels = 10 ** np.linspace(0, np.log10(zz.max() + 1), 22) - 1
    ax.contourf(xx, yy, zz, levels=levels, cmap="Blues", alpha=0.18)
    ax.contour(xx, yy, zz, levels=levels, colors="steelblue", linewidths=0.4, alpha=0.4)
    ax.scatter(MINIMA[:, 0], MINIMA[:, 1], marker="*", s=260, color="gold", edgecolors="black", linewidths=0.8, zorder=20)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    trajs, colors = trajectories()
    n_vis = trajs[0].shape[0]

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    frames = 220

    def draw(i):
        r = (i + 1) / frames
        n = max(2, int(round((1 - (1 - r) ** 1.6) * n_vis)))
        ax.clear()
        draw_landscape(ax)
        for traj, c in zip(trajs, colors):
            ax.plot(traj[:n, 0], traj[:n, 1], color=c, lw=1.4, alpha=0.9, zorder=10)
            ax.plot(traj[n - 1, 0], traj[n - 1, 1], "o", color=c, ms=3.2, zorder=11)
        add_watermark(fig)
        return []

    draw(frames - 1)
    fig.savefig(OPTIM_VIS / "himmelblau_flow_poster.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    FuncAnimation(fig, draw, frames=frames, interval=1000 / 60, blit=False).save(OPTIM_VIS / "himmelblau_flow.mp4", writer=make_writer(60), dpi=170)
    plt.close(fig)
    write_page("himmelblau_flow", TITLE, DESCRIPTION, __file__)
    print("done himmelblau_flow")


if __name__ == "__main__":
    main()
