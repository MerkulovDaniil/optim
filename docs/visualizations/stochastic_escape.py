"""Gradient flow vs stochastic (Langevin) flow on a double-well potential.

The deterministic gradient flow gets trapped in whatever basin it starts in.
The stochastic flow dx = -f'(x) dt + sigma dW keeps exploring, crosses the
barrier and its histogram relaxes to the Gibbs density rho* ∝ exp(-2f/sigma^2),
which puts most mass on the *global* minimum.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from flows_common import OPTIM_VIS, add_watermark, make_writer, write_page

GF_BLUE = "#2E6FE0"
SDE_PURPLE = "#8B5CF6"

TITLE = "Escaping a local minimum: GF vs SDE"
DESCRIPTION = r"""
On a double-well potential the deterministic **gradient flow**
$\dot x=-f'(x)$ is trapped in its starting basin. The **stochastic flow**
$$
dx(t) = -f'\bigl(x(t)\bigr)\,dt + \sigma\,dW(t)
$$
keeps exploring: it climbs over the barrier and its density relaxes to the
**Gibbs distribution** $\rho^*(x)\propto e^{-2f(x)/\sigma^2}$, concentrated on the
global minimum. The bottom bars track the fraction of trajectories that have
reached the global well.
"""


def f(x):
    return (x**2 - 1.0) ** 2 + 0.15 * x


def fprime(x):
    return 4.0 * x * (x**2 - 1.0) + 0.15


def simulate():
    rng = np.random.default_rng(7)
    n, steps, dt, sigma = 500, 1000, 0.012, 0.9
    x0 = 1.0 + 0.05 * rng.standard_normal(n)
    gf = np.empty((steps + 1, n))
    sde = np.empty((steps + 1, n))
    gf[0] = x0
    sde[0] = x0.copy()
    for k in range(steps):
        gf[k + 1] = gf[k] - dt * fprime(gf[k])
        sde[k + 1] = sde[k] - dt * fprime(sde[k]) + sigma * np.sqrt(dt) * rng.standard_normal(n)
        sde[k + 1] = np.clip(sde[k + 1], -2.1, 2.1)
    return gf, sde, sigma


def main():
    gf, sde, sigma = simulate()
    steps = gf.shape[0] - 1
    xs = np.linspace(-2.0, 2.0, 600)
    fx = f(xs)
    rho = np.exp(-2.0 * fx / sigma**2)
    rho /= np.trapezoid(rho, xs)
    frac_gf = np.mean(gf < 0.0, axis=1)
    frac_sde = np.mean(sde < 0.0, axis=1)

    fig = plt.figure(figsize=(8.2, 6.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.0, 0.6], hspace=0.42, left=0.10, right=0.96, top=0.95, bottom=0.08)
    axF, axD, axB = (fig.add_subplot(gs[i]) for i in range(3))
    frames = 230

    def draw(i):
        r = (i + 1) / frames
        k = min(steps, int(round(r * steps)))
        for ax in (axF, axD, axB):
            ax.clear()

        # --- potential + particles ---
        axF.plot(xs, fx, color="#222633", lw=2.0, zorder=3)
        axF.fill_between(xs, fx, fx.max(), color=SDE_PURPLE, alpha=0.05)
        axF.scatter(sde[k], f(sde[k]), s=10, color=SDE_PURPLE, alpha=0.30, zorder=4, edgecolors="none")
        axF.scatter(gf[k], f(gf[k]), s=12, color=GF_BLUE, alpha=0.55, zorder=5, edgecolors="none")
        axF.set_xlim(-2, 2)
        axF.set_ylim(-0.5, 4.0)
        axF.set_yticks([])
        axF.tick_params(labelsize=7)
        axF.text(0.02, 0.92, "GF", transform=axF.transAxes, color=GF_BLUE, fontsize=14, fontweight="bold")
        axF.text(0.93, 0.92, "SDE", transform=axF.transAxes, color=SDE_PURPLE, fontsize=14, fontweight="bold", ha="right")
        for s in ("top", "right"):
            axF.spines[s].set_visible(False)

        # --- density vs Gibbs ---
        axD.plot(xs, rho, "--", color="#222633", lw=1.6, zorder=5)
        axD.hist(sde[k], bins=40, range=(-2, 2), density=True, color=SDE_PURPLE, alpha=0.55)
        axD.hist(gf[k], bins=40, range=(-2, 2), density=True, color=GF_BLUE, alpha=0.55)
        axD.set_xlim(-2, 2)
        axD.set_ylim(0, max(rho.max() * 1.6, 1.2))
        axD.set_yticks([])
        axD.tick_params(labelsize=7)
        axD.text(0.5, 0.86, r"$\rho^*(x)\propto e^{-2f/\sigma^2}$", transform=axD.transAxes, ha="center", fontsize=10, color="#222633")
        for s in ("top", "right"):
            axD.spines[s].set_visible(False)

        # --- fraction at global min ---
        axB.bar([0, 1], [frac_gf[k], frac_sde[k]], color=[GF_BLUE, SDE_PURPLE], alpha=0.8, width=0.55)
        axB.set_xticks([0, 1])
        axB.set_xticklabels(["GF", "SDE"], fontsize=9)
        axB.set_ylim(0, 1)
        axB.set_ylabel("at global min", fontsize=8)
        axB.axhline(0.5, color="gray", lw=0.7, ls=":")
        axB.tick_params(labelsize=7)
        for s in ("top", "right"):
            axB.spines[s].set_visible(False)
        add_watermark(fig)
        return []

    draw(frames - 1)
    fig.savefig(OPTIM_VIS / "stochastic_escape_poster.pdf", format="pdf", bbox_inches="tight", pad_inches=0.03)
    FuncAnimation(fig, draw, frames=frames, interval=1000 / 60, blit=False).save(OPTIM_VIS / "stochastic_escape.mp4", writer=make_writer(60), dpi=160)
    plt.close(fig)
    write_page("stochastic_escape", TITLE, DESCRIPTION, __file__)
    print("done stochastic_escape")


if __name__ == "__main__":
    main()
