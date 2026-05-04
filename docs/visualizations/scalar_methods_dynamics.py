from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from continuous_time_common import COLORS, LECTURES, OPTIM_VIS, save_pdf


def f(x):
    return 0.35 * np.exp(-2.0 * (x - 4.5) ** 2) + 1.6 * np.exp(-4.5 * (x + 2.8) ** 2) + 2.0 * np.exp(-0.45 * (x - 2.1) ** 4) + 0.08 * np.cos(18 * x) - np.tanh(0.55 * x**2) - 0.055 * np.abs(x)


def grad(x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def simulate():
    rng = np.random.default_rng(3)
    methods = ["GF", "HB", "NAG"]
    colors = {"GF": COLORS["gf"], "HB": COLORS["hb"], "NAG": COLORS["agf"]}
    n, steps, dt, sigma = 180, 520, 0.035, 0.18
    x0 = rng.uniform(-5.6, 5.6, size=n)
    states = {m: np.zeros((steps + 1, n)) for m in methods}
    velocities = {m: np.zeros(n) for m in methods}
    for m in methods:
        states[m][0] = x0
    for k in range(steps):
        noise = rng.normal(size=n)
        states["GF"][k + 1] = states["GF"][k] - dt * grad(states["GF"][k]) + np.sqrt(2 * sigma * dt) * noise
        velocities["HB"] = 0.88 * velocities["HB"] - dt * grad(states["HB"][k]) + 0.35 * np.sqrt(2 * sigma * dt) * noise
        states["HB"][k + 1] = states["HB"][k] + velocities["HB"]
        t = (k + 1) * dt
        velocities["NAG"] += dt * (-3.0 / max(t, 0.16) * velocities["NAG"] - grad(states["NAG"][k])) + 0.18 * np.sqrt(2 * sigma * dt) * noise
        states["NAG"][k + 1] = states["NAG"][k] + dt * velocities["NAG"]
        for m in methods:
            states[m][k + 1] = np.clip(states[m][k + 1], -6.2, 6.2)
    return states, colors


def draw_panel(ax, xs, yn, method, points, color, title_suffix=""):
    ax.set_facecolor("#FBFBFD")
    ax.fill_between(xs, 0, yn, color="#E0E7FF", alpha=0.40)
    ax.plot(xs, yn, color="#111827", lw=2.0)
    y = np.interp(points, xs, yn) + 0.035 * np.sin(np.arange(len(points)) * 1.7)
    ax.scatter(points, y, s=16, color=color, alpha=0.72, edgecolor="white", linewidth=0.35)
    ax.text(-6.05, 1.08, method + title_suffix, color=color, fontsize=14, weight="bold")
    ax.set_xlim(-6.2, 6.2)
    ax.set_ylim(-0.06, 1.18)
    ax.axis("off")


def make_poster(states, colors):
    xs = np.linspace(-6.2, 6.2, 1000)
    vals = f(xs)
    yn = (vals - vals.min()) / (vals.max() - vals.min())
    fig, axes = plt.subplots(3, 1, figsize=(13.4, 6.2), facecolor="white", sharex=True)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.985, bottom=0.04, hspace=0.07)
    for ax, method in zip(axes, ["GF", "HB", "NAG"]):
        draw_panel(ax, xs, yn, method, states[method][-1], colors[method])
    save_pdf(fig, LECTURES / "scalar_methods_dynamics.pdf")


def make_video(states, colors):
    xs = np.linspace(-6.2, 6.2, 1000)
    vals = f(xs)
    yn = (vals - vals.min()) / (vals.max() - vals.min())
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 8.0), facecolor="white", sharex=True)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.88, bottom=0.055, hspace=0.08)
    title = fig.text(0.03, 0.958, "Scalar dynamics", fontsize=18, weight="bold", color="#0F172A")
    subtitle = fig.text(0.03, 0.922, "GF / HB / NAG", fontsize=11, color="#64748B")
    frames = 140

    def update(i):
        idx = min(states["GF"].shape[0] - 1, int(i / (frames - 1) * (states["GF"].shape[0] - 1)))
        for ax, method in zip(axes, ["GF", "HB", "NAG"]):
            ax.clear()
            draw_panel(ax, xs, yn, method, states[method][idx], colors[method])
        subtitle.set_text("GF / HB / NAG")
        return [title, subtitle]

    out = OPTIM_VIS / "scalar_methods_dynamics.mp4"
    writer = FFMpegWriter(fps=30, codec="h264", bitrate=-1, extra_args=["-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium"])
    FuncAnimation(fig, update, frames=frames, interval=1000 / 30, blit=False).save(out, writer=writer, dpi=170)
    plt.close(fig)
    print(f"saved {out}")


def write_optim_page():
    page = OPTIM_VIS / "scalar_methods_dynamics.md"
    page.write_text(
        """---
title: "Scalar Dynamics: GF / HB / NAG"
---

:::{.video}
scalar_methods_dynamics.mp4
:::

[Code](scalar_methods_dynamics.py)
""",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), OPTIM_VIS / "scalar_methods_dynamics.py")
    print(f"saved {page}")


def main():
    states, colors = simulate()
    make_poster(states, colors)
    make_video(states, colors)
    write_optim_page()


if __name__ == "__main__":
    main()
