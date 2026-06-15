"""Lyapunov-energy animations for the accelerated gradient flow.

Two regimes share the same machinery:
  - convex:  X'' + (3/t) X' + grad f = 0,  E = t^2 (f-f*) + 2||X - x* + (t/2)X'||^2
  - strongly convex: X'' + 2 sqrt(mu) X' + grad f = 0,  E = (f-f*) + 1/2 ||X' + sqrt(mu)(X-x*)||^2
Left panel: the trajectory on the landscape. Right panels: f(X(t))-f* with its
theoretical envelope, and the Lyapunov energy E(t) that certifies the rate.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from scipy.integrate import solve_ivp

from flows_common import OPTIM_VIS, add_watermark, make_writer, write_page

PURPLE = "#6C4CF1"
RED = "#E1322B"


def _problem(mode):
    theta = np.deg2rad(25.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mu, L = 1.0, 9.0
    A = rot @ np.diag([mu, L]) @ rot.T
    x0 = np.array([2.6, 1.9])

    def grad(x):
        return A @ x

    if mode == "convex":
        t0, T, n = 0.05, 15.0, 1700

        def rhs(t, s):
            x, v = s[:2], s[2:]
            return np.r_[v, -3.0 / t * v - grad(x)]
    else:
        t0, T, n = 0.0, 8.0, 1500
        damp = 2.0 * np.sqrt(mu)

        def rhs(t, s):
            x, v = s[:2], s[2:]
            return np.r_[v, -damp * v - grad(x)]

    t = np.linspace(max(t0, 1e-3), T, n)
    sol = solve_ivp(rhs, (t[0], t[-1]), np.r_[x0, [0.0, 0.0]], t_eval=t, rtol=1e-10, atol=1e-12, max_step=0.01)
    X = sol.y[:2].T
    V = sol.y[2:].T
    fval = 0.5 * np.einsum("ij,jk,ik->i", X, A, X)
    if mode == "convex":
        w = X + (t[:, None] / 2.0) * V
        E = t**2 * fval + 2.0 * np.sum(w**2, axis=1)
        env = E[0] / t**2
    else:
        z = V + np.sqrt(mu) * X
        E = fval + 0.5 * np.sum(z**2, axis=1)
        env = E[0] * np.exp(-np.sqrt(mu) * t)
    return {"A": A, "x0": x0, "t": t, "X": X, "f": np.maximum(fval, 1e-9), "E": np.maximum(E, 1e-9), "env": env, "mode": mode}


def _draw_landscape(ax, prob):
    X = prob["X"]
    pad = 0.6
    xlim = (min(X[:, 0].min(), 0) - pad, max(X[:, 0].max(), 0) + pad)
    ylim = (min(X[:, 1].min(), 0) - pad, max(X[:, 1].max(), 0) + pad)
    xg = np.linspace(*xlim, 300)
    yg = np.linspace(*ylim, 300)
    xx, yy = np.meshgrid(xg, yg)
    A = prob["A"]
    zz = 0.5 * (A[0, 0] * xx**2 + 2 * A[0, 1] * xx * yy + A[1, 1] * yy**2)
    zmin = max(float(zz[zz > 0].min()), 1e-4)
    levels = np.geomspace(zmin, float(zz.max()), 22)
    ax.contour(xx, yy, zz + zmin, levels=levels, colors="#9AA3B2", norm=LogNorm(vmin=zmin), linewidths=0.7, alpha=0.7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def build(mode):
    prob = _problem(mode)
    fig = plt.figure(figsize=(11.2, 5.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1, 1], wspace=0.26, hspace=0.42, left=0.02, right=0.965, top=0.93, bottom=0.12)
    axL = fig.add_subplot(gs[:, 0])
    axT = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, 1])
    return prob, fig, (axL, axT, axB)


def _style_curves(prob, axT, axB):
    t = prob["t"]
    tag = "AGF" if prob["mode"] == "convex" else "SC-AGF"
    if prob["mode"] == "convex":
        axT.set_xscale("log")
        axB.set_xscale("log")
        ref_label = r"$\mathcal{O}(1/t^2)$"
    else:
        ref_label = r"$\mathcal{O}(e^{-\sqrt{\mu}\,t})$"
    axT.set_yscale("log")
    axB.set_yscale("log")
    axT.set_xlim(t[0], t[-1])
    axB.set_xlim(t[0], t[-1])
    axT.set_ylim(min(prob["f"].min(), prob["env"].min()) * 0.5, prob["f"].max() * 2)
    axB.set_ylim(prob["E"].min() * 0.5, prob["E"].max() * 2)
    axT.set_ylabel(r"$f(X(t))-f^*$", fontsize=9)
    axB.set_ylabel(r"$E(t)$", fontsize=9, color=RED)
    axB.set_xlabel(r"time $t$", fontsize=9)
    for ax in (axT, axB):
        ax.grid(alpha=0.25, which="both", linestyle=":")
        ax.tick_params(labelsize=7)
    return tag, ref_label


def render(stem, title, description, mode, *, frames=220, fps=60):
    prob, fig, (axL, axT, axB) = build(mode)
    t, X, fv, E, env = prob["t"], prob["X"], prob["f"], prob["E"], prob["env"]
    tag, ref_label = _style_curves(prob, axT, axB)
    n_total = len(t)

    def frame(i):
        r = (i + 1) / frames
        n = max(2, int(round((1 - (1 - r) ** 1.8) * n_total)))
        for ax in (axL, axT, axB):
            ax.clear()
        _draw_landscape(axL, prob)
        axL.plot(X[:n, 0], X[:n, 1], color=PURPLE, lw=2.2, zorder=10)
        axL.plot(X[n - 1, 0], X[n - 1, 1], "o", color=PURPLE, ms=6, zorder=11)
        axL.plot(X[0, 0], X[0, 1], "o", color="#7A879B", ms=6, zorder=11)
        axL.plot(0, 0, marker="*", color="gold", ms=16, markeredgecolor="black", markeredgewidth=0.7, zorder=12)
        axL.text(0.02, 0.97, tag, transform=axL.transAxes, color=PURPLE, fontsize=14, fontweight="bold", va="top")

        _style_curves(prob, axT, axB)
        axT.plot(t, env, "--", color="#8A8F98", lw=1.1)
        axT.text(0.97, 0.9, ref_label, transform=axT.transAxes, ha="right", color="#8A8F98", fontsize=8)
        axT.plot(t[:n], fv[:n], color=PURPLE, lw=1.9)
        axT.plot(t[n - 1], fv[n - 1], "o", color=PURPLE, ms=4.5)
        axB.plot(t[:n], E[:n], color=RED, lw=1.9)
        axB.plot(t[n - 1], E[n - 1], "o", color=RED, ms=4.5)
        add_watermark(fig)
        return []

    # poster = final frame
    frame(frames - 1)
    fig.savefig(OPTIM_VIS / f"{stem}_poster.pdf", format="pdf", bbox_inches="tight", pad_inches=0.03)
    FuncAnimation(fig, frame, frames=frames, interval=1000 / fps, blit=False).save(OPTIM_VIS / f"{stem}.mp4", writer=make_writer(fps), dpi=160)
    plt.close(fig)
    write_page(stem, title, description, _caller_file(stem))
    print(f"done {stem}")


def _caller_file(stem):
    # the wrapper scripts live next to this module
    return str(OPTIM_VIS / f"{stem}.py")
