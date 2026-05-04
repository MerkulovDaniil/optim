from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import LogNorm

from continuous_time_common import COLORS, LECTURES, OPTIM_VIS, quadratic_problem, rosenbrock_problem


METHODS = ["GF", "AGF", "GD"]


def draw_single(ax, problem, method, ratio=1.0, *, row_label=None):
    xx, yy, zz = problem["grid"]
    zmin = max(float(zz[zz > 0].min()), 1e-4)
    levels = np.geomspace(zmin, float(zz.max()), 17)
    ax.contour(xx, yy, zz + zmin, levels=levels, cmap="BuPu", norm=LogNorm(vmin=zmin), linewidths=0.85, alpha=0.9)
    traj = problem["trajs"][method]
    n = max(2, int(ratio * len(traj)))
    color = problem["colors"][method]
    ax.plot(traj[:n, 0], traj[:n, 1], color=color, lw=2.2, zorder=10)
    ax.plot(traj[n - 1, 0], traj[n - 1, 1], "o", color=color, ms=5.0, zorder=11)
    ax.plot(traj[0, 0], traj[0, 1], "ko", ms=3.7, zorder=12)
    ax.plot(*problem["minimizer"], marker="*", color="gold", ms=9.5, zorder=12)
    ax.set_xlim(*problem["xlim"])
    ax.set_ylim(*problem["ylim"])
    ax.set_aspect("equal")
    ax.grid(alpha=0.30, linestyle=":")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    if row_label:
        ax.set_ylabel(row_label, fontsize=10)
    else:
        ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title(method, fontsize=10, color=color, fontweight="bold")


def make_figure(problems, ratio=1.0, *, video=False):
    if video:
        fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8))
        fig.subplots_adjust(left=0.045, right=0.985, top=0.90, bottom=0.055, wspace=0.055, hspace=0.19)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.5), layout="tight")
    row_labels = [r"quadratic $\kappa=80$", "Rosenbrock"]
    for row, problem in enumerate(problems):
        for col, method in enumerate(METHODS):
            draw_single(axes[row, col], problem, method, ratio, row_label=row_labels[row] if col == 0 else None)
    fig.suptitle("GF / AGF / GD", fontsize=13, fontweight="bold")
    fig.text(0.985, 0.015, "@fminxyz", ha="right", va="bottom", fontsize=8, color="gray", alpha=0.65)
    return fig, axes


def make_poster(problems):
    fig, _axes = make_figure(problems, ratio=1.0)
    for path in [LECTURES / "continuous_time_flows_poster.pdf", OPTIM_VIS / "continuous_time_flows_poster.pdf"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        print(f"saved {path}")
    plt.close(fig)


def make_video(problems):
    fig, axes = make_figure(problems, ratio=0.01, video=True)
    frames = 170

    def update(i):
        ratio = (i + 1) / frames
        eased = 1 - (1 - ratio) ** 2.0
        row_labels = [r"quadratic $\kappa=80$", "Rosenbrock"]
        for row, problem in enumerate(problems):
            for col, method in enumerate(METHODS):
                ax = axes[row, col]
                ax.clear()
                draw_single(ax, problem, method, eased, row_label=row_labels[row] if col == 0 else None)
        return axes.ravel().tolist()

    out = OPTIM_VIS / "continuous_time_flows.mp4"
    writer = FFMpegWriter(
        fps=60,
        metadata={"artist": "@fminxyz"},
        bitrate=-1,
        codec="h264",
        extra_args=["-preset", "ultrafast", "-crf", "25", "-pix_fmt", "yuv420p", "-tune", "animation"],
    )
    FuncAnimation(fig, update, frames=frames, interval=1000 / 60, blit=False).save(out, writer=writer, dpi=180)
    plt.close(fig)
    print(f"saved {out}")


def write_optim_page():
    page = OPTIM_VIS / "continuous_time_flows.md"
    page.write_text(
        """---
title: "GF / AGF / GD"
---

:::{.video}
continuous_time_flows.mp4
:::

[Code](continuous_time_flows.py)
""",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), OPTIM_VIS / "continuous_time_flows.py")
    print(f"saved {page}")


def main():
    OPTIM_VIS.mkdir(parents=True, exist_ok=True)
    problems = [quadratic_problem(), rosenbrock_problem()]
    make_poster(problems)
    make_video(problems)
    write_optim_page()


if __name__ == "__main__":
    main()
