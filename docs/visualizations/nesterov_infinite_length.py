from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from continuous_time_common import COLORS, DATA, OPTIM_VIS


def main():
    orbit = np.genfromtxt(DATA / "nesterov_static_orbit_eps50.tsv", names=True, delimiter="\t")
    series = np.genfromtxt(DATA / "nesterov_static_series_eps50.tsv", names=True, delimiter="\t")
    x1, x2, t = orbit["x1"], orbit["x2"], orbit["time"]
    ts, length, fval = series["time"], series["arclength"], series["f_value"]
    a, eps = 0.02, 50.0

    def f_core(r):
        out = np.zeros_like(r, dtype=float)
        mask = (r > 0) & (r <= np.exp(-2))
        out[mask] = r[mask] / (-np.log(r[mask]))
        out[r > np.exp(-2)] = np.exp(-2) / 2 + 0.5 * (r[r > np.exp(-2)] - np.exp(-2))
        return out

    xx, yy = np.meshgrid(np.linspace(-0.047, 0.047, 220), np.linspace(-0.047, 0.047, 220))
    pot = f_core(np.sqrt(xx**2 + yy**2)) + eps * 0.5 * np.maximum(xx - a, 0) ** 2
    levels = np.geomspace(max(pot[pot > 0].min(), 1e-7), pot.max(), 20)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 5.6), layout="tight", gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax0.set_xlim(-0.046, 0.046)
    ax0.set_ylim(-0.046, 0.046)
    ax0.set_aspect("equal")
    ax0.contourf(xx, yy, np.log10(pot + 1e-8), levels=22, cmap="PuBu", alpha=0.62)
    ax0.contour(xx, yy, pot, levels=levels[::2], colors="#475569", linewidths=0.36, alpha=0.36)
    ax0.axvline(a, color="#0F172A", linestyle="--", linewidth=0.9, alpha=0.40)
    ax0.grid(alpha=0.35, linestyle=":")
    ax0.set_xlabel(r"$x_1$", fontsize=10)
    ax0.set_ylabel(r"$x_2$", fontsize=10)
    ax0.set_title("Ryu 2026: orbit", fontsize=10)
    ax0.plot(0, 0, marker="*", color="gold", ms=11, zorder=4)
    path_line, = ax0.plot([], [], color=COLORS["agf"], lw=1.9)
    dot, = ax0.plot([], [], "o", ms=5.4, color=COLORS["gf"])

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(max(ts[1], 1e-3), ts[-1])
    ax1.set_ylim(min(fval[fval > 0].min(), length[length > 0].min()) * 0.8, max(length.max(), fval.max()) * 1.25)
    ax1.grid(alpha=0.35, which="both", linestyle=":")
    ax1.set_title("value vs path length", fontsize=10)
    ax1.set_xlabel("time", fontsize=10)
    val_line, = ax1.plot([], [], color=COLORS["agf"], lw=1.9, label=r"$f(X(t))$")
    len_line, = ax1.plot([], [], color=COLORS["gf"], lw=1.9, label="length")
    ax1.legend(fontsize=8, frameon=False)
    fig.suptitle("Nesterov flow: convergence without rectifiability", fontsize=13, fontweight="bold")
    fig.text(0.985, 0.015, "@fminxyz", ha="right", va="bottom", fontsize=8, color="gray", alpha=0.65)

    frames = 240

    def update(i):
        ratio = i / (frames - 1)
        idx = min(len(t) - 1, max(2, int(ratio**0.62 * (len(t) - 1))))
        j = min(len(ts) - 1, max(2, int(ratio**0.62 * (len(ts) - 1))))
        path_line.set_data(x1[:idx], x2[:idx])
        dot.set_data([x1[idx]], [x2[idx]])
        val_line.set_data(ts[:j], fval[:j])
        len_line.set_data(ts[:j], length[:j] + 1e-14)
        return [path_line, dot, val_line, len_line]

    out = OPTIM_VIS / "nesterov_infinite_length.mp4"
    writer = FFMpegWriter(fps=60, metadata={"artist": "@fminxyz"}, bitrate=-1, codec="h264", extra_args=["-preset", "ultrafast", "-crf", "25", "-pix_fmt", "yuv420p", "-tune", "animation"])
    FuncAnimation(fig, update, frames=frames, interval=1000 / 60, blit=False).save(out, writer=writer, dpi=190)
    plt.close(fig)
    print(f"saved {out}")

    page = OPTIM_VIS / "nesterov_infinite_length.md"
    page.write_text(
        """---
title: "Nesterov Flow: Infinite Path Length"
---

:::{.video}
nesterov_infinite_length.mp4
:::

[Code](nesterov_infinite_length.py)
""",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), OPTIM_VIS / "nesterov_infinite_length.py")
    print(f"saved {page}")


if __name__ == "__main__":
    main()
