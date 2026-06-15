"""Shared helpers for the continuous-time / discrete-method flow visualizations.

Used by gd_vs_gf, gd_vs_ppm, gd_vs_ppm_optimal and gd_gf_nag_hb. Keeps the
landscape, the dashed-contour style, the 60 fps writer and a generic
"progressively draw trajectories with a gliding marker" animator in one place
so the individual scripts stay short.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import LogNorm
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from continuous_time_common import COLORS, OPTIM_VIS  # noqa: F401


def make_writer(fps: int = 60) -> FFMpegWriter:
    """h264 / yuv420p writer tuned for smooth flat-color animation."""
    return FFMpegWriter(
        fps=fps,
        metadata={"artist": "@fminxyz"},
        bitrate=-1,
        codec="h264",
        extra_args=["-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-tune", "animation"],
    )


def add_watermark(fig) -> None:
    fig.text(0.99, 0.012, "@fminxyz", ha="right", va="bottom", fontsize=8, color="gray", alpha=0.6)


def resample_by_arclength(points: np.ndarray, n_out: int) -> np.ndarray:
    """points: (N, 2) -> (n_out, 2) sampled uniformly along the polyline."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-12:
        return np.repeat(points[:1], n_out, axis=0)
    targets = np.linspace(0.0, total, n_out)
    x = np.interp(targets, cum, points[:, 0])
    y = np.interp(targets, cum, points[:, 1])
    return np.column_stack([x, y])


def smooth_through(points: np.ndarray, n_out: int = 400) -> np.ndarray:
    """Smooth cubic curve through the given iterates (parameterized by chord length)."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return resample_by_arclength(points, n_out)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    t /= t[-1]
    cs_x = CubicSpline(t, points[:, 0], bc_type="natural")
    cs_y = CubicSpline(t, points[:, 1], bc_type="natural")
    tt = np.linspace(0.0, 1.0, n_out)
    return np.column_stack([cs_x(tt), cs_y(tt)])


# ---------------------------------------------------------------------------
# Ill-conditioned rotated quadratic shared by the GD/GF/PPM scenes.
# f(x) = 1/2 x^T A x, minimizer at the origin.
# Eigenvalues chosen so that the optimal GD step 2/(mu+L) ~= 5.97.
# ---------------------------------------------------------------------------
def spiral_problem():
    theta = np.deg2rad(-28.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mu, L = 0.060, 0.275
    A = rot @ np.diag([mu, L]) @ rot.T
    x0 = np.array([-2.55, 1.95])

    def grad(x):
        return A @ x

    # gradient flow: x' = -A x
    sol = solve_ivp(lambda _t, x: -grad(x), (0.0, 90.0), x0, t_eval=np.linspace(0.0, 90.0, 1400), rtol=1e-9, atol=1e-11)
    gf = resample_by_arclength(sol.y.T, 600)

    alpha_gd = 2.0 / (mu + L)            # ~= 5.97  (optimal worst-case GD step)
    alpha_ppm = 2.0 * alpha_gd           # ~= 11.94 larger step PPM stays stable

    def run_gd(alpha, n=9):
        x = x0.copy()
        pts = [x.copy()]
        for _ in range(n):
            x = x - alpha * grad(x)
            pts.append(x.copy())
        return np.asarray(pts)

    def run_ppm(alpha, n=7):
        # implicit Euler / proximal point on the quadratic: x <- (I + alpha A)^{-1} x
        M = np.linalg.inv(np.eye(2) + alpha * A)
        x = x0.copy()
        pts = [x.copy()]
        for _ in range(n):
            x = M @ x
            pts.append(x.copy())
        return np.asarray(pts)

    xg = np.linspace(-3.2, 3.4, 320)
    yg = np.linspace(-2.4, 2.3, 320)
    xx, yy = np.meshgrid(xg, yg)
    zz = 0.5 * (A[0, 0] * xx**2 + 2 * A[0, 1] * xx * yy + A[1, 1] * yy**2)

    return {
        "A": A,
        "x0": x0,
        "xlim": (-3.2, 3.4),
        "ylim": (-2.4, 2.3),
        "grid": (xx, yy, zz),
        "minimizer": (0.0, 0.0),
        "alpha_gd": alpha_gd,
        "alpha_ppm": alpha_ppm,
        "gf": gf,
        "gd": run_gd(alpha_gd),
        "gd_iter": run_gd,
        "ppm_iter": run_ppm,
    }


def draw_contours(ax, prob) -> None:
    xx, yy, zz = prob["grid"]
    zmin = max(float(zz[zz > 0].min()), 1e-4)
    levels = np.geomspace(zmin, float(zz.max()), 26)
    ax.contour(xx, yy, zz + zmin, levels=levels, cmap="cool", norm=LogNorm(vmin=zmin), linewidths=0.7, linestyles="--", alpha=0.55)
    ax.set_xlim(*prob["xlim"])
    ax.set_ylim(*prob["ylim"])
    ax.set_aspect("equal")
    ax.axis("off")


def _prepare(specs, n_dense=600):
    """Each spec: dict(points=(N,2), color, markers=bool, smooth=bool, label).

    smooth=True draws a cubic curve through the iterates (PPM, flows); otherwise
    the polyline is followed segment by segment (GD zig-zag). vertices are always
    the original iterates so markers land on them.
    """
    out = []
    for s in specs:
        pts = np.asarray(s["points"], dtype=float)
        dense = smooth_through(pts, n_dense) if s.get("smooth") else resample_by_arclength(pts, n_dense)
        out.append({**s, "vertices": pts, "dense": dense})
    return out


def render_trajectory_video(out_path: Path, prob, specs, *, corner_labels=None, title=None, frames=240, fps=60, figsize=(9.6, 5.4)):
    import matplotlib.pyplot as plt

    prepared = _prepare(specs)
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    def draw(progress):
        ax.clear()
        draw_contours(ax, prob)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", color=COLORS["ink"])
        # start point + minimizer (always shown)
        ax.plot(prob["x0"][0], prob["x0"][1], "o", color="#7A879B", ms=8, zorder=5)
        ax.plot(*prob["minimizer"], marker="*", color="gold", ms=17, markeredgecolor="black", markeredgewidth=0.7, zorder=12)
        for s in prepared:
            dense = s["dense"]
            n = max(2, int(round(progress * len(dense))))
            seg = dense[:n]
            ax.plot(seg[:, 0], seg[:, 1], color=s["color"], lw=2.4, zorder=10, solid_capstyle="round")
            if s.get("markers"):
                # show iterate vertices that are already revealed
                revealed = s["vertices"][: 1 + int(round(progress * (len(s["vertices"]) - 1)))]
                ax.plot(revealed[:, 0], revealed[:, 1], "o", color=s["color"], ms=5.0, zorder=11, alpha=0.9)
            ax.plot(seg[-1, 0], seg[-1, 1], "o", color=s["color"], ms=7.0, zorder=13)
        if corner_labels:
            for (txt, color, xy) in corner_labels:
                ax.text(xy[0], xy[1], txt, transform=ax.transAxes, color=color, fontsize=14, fontweight="bold", ha="left", va="top")
        add_watermark(fig)

    def update(i):
        r = (i + 1) / frames
        eased = 1 - (1 - r) ** 2.2
        draw(eased)
        return []

    FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False).save(out_path, writer=make_writer(fps), dpi=170)
    plt.close(fig)
    print(f"saved {out_path}")


def render_trajectory_poster(path: Path, prob, specs, *, corner_labels=None, title=None, figsize=(9.6, 5.4)):
    import matplotlib.pyplot as plt

    prepared = _prepare(specs)
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    draw_contours(ax, prob)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=COLORS["ink"])
    ax.plot(prob["x0"][0], prob["x0"][1], "o", color="#7A879B", ms=8, zorder=5)
    ax.plot(*prob["minimizer"], marker="*", color="gold", ms=17, markeredgecolor="black", markeredgewidth=0.7, zorder=12)
    for s in prepared:
        dense = s["dense"]
        ax.plot(dense[:, 0], dense[:, 1], color=s["color"], lw=2.4, zorder=10, solid_capstyle="round")
        if s.get("markers"):
            ax.plot(s["vertices"][:, 0], s["vertices"][:, 1], "o", color=s["color"], ms=5.0, zorder=11, alpha=0.9)
    if corner_labels:
        for (txt, color, xy) in corner_labels:
            ax.text(xy[0], xy[1], txt, transform=ax.transAxes, color=color, fontsize=14, fontweight="bold", ha="left", va="top")
    add_watermark(fig)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {path}")


def write_page(stem: str, title: str, description: str, script_file: str) -> None:
    """Write NAME.md (rich page) and copy NAME.py next to it."""
    import shutil

    page = OPTIM_VIS / f"{stem}.md"
    body = f"""---
title: "{title}"
---

{description.strip()}

:::{{.video}}
{stem}.mp4
:::

[Code]({stem}.py)
"""
    page.write_text(body, encoding="utf-8")
    src = Path(script_file).resolve()
    dst = OPTIM_VIS / f"{stem}.py"
    if src != dst:
        shutil.copy2(src, dst)
    print(f"saved {page}")
