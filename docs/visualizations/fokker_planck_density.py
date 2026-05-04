from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from continuous_time_common import COLORS, LECTURES, OPTIM_VIS, save_pdf


def potential(x):
    return 0.25 * (x**2 - 1.0) ** 2 + 0.09 * x + 0.018 * np.sin(7 * x) * np.exp(-0.25 * x**2)


def grad_potential(x):
    return x * (x**2 - 1.0) + 0.09 + 0.018 * np.exp(-0.25 * x**2) * (7 * np.cos(7 * x) - 0.5 * x * np.sin(7 * x))


def simulate():
    rng = np.random.default_rng(7)
    sigma, dt, total_time, n_particles = 0.72, 0.0025, 8.0, 5200
    x = rng.normal(-1.65, 0.20, size=n_particles)
    bins = np.linspace(-2.55, 2.55, 155)
    centers = 0.5 * (bins[:-1] + bins[1:])
    xs = np.linspace(-2.55, 2.55, 500)
    stat = np.exp(-2 * potential(xs) / sigma**2)
    stat /= np.trapezoid(stat, xs)
    stat_centers = np.interp(centers, xs, stat)
    stat_centers /= np.trapezoid(stat_centers, centers)
    times, densities, particles = [], [], []
    snapshot_times = [0.0, 0.45, 1.4, 3.5, 8.0]
    snapshots, next_snap = {}, 0

    for k in range(int(total_time / dt) + 1):
        cur_t = k * dt
        if k % 24 == 0:
            hist, _ = np.histogram(x, bins=bins, density=True)
            densities.append(hist)
            times.append(cur_t)
            particles.append(rng.choice(x, size=360, replace=False))
        if next_snap < len(snapshot_times) and cur_t >= snapshot_times[next_snap] - 0.5 * dt:
            snapshots[snapshot_times[next_snap]] = x.copy()
            next_snap += 1
        if k < int(total_time / dt):
            x += -grad_potential(x) * dt + sigma * np.sqrt(dt) * rng.normal(size=n_particles)
            x = np.clip(x, -3.2, 3.2)

    dens = np.asarray(densities)
    times = np.asarray(times)
    dx = centers[1] - centers[0]
    q = stat_centers / (np.sum(stat_centers) * dx)
    kl, wd = [], []
    q_cdf = np.cumsum(q) * dx
    for row in dens:
        p = row / (np.sum(row) * dx)
        mask = (p > 0) & (q > 0)
        kl.append(np.sum(p[mask] * np.log(p[mask] / q[mask])) * dx)
        wd.append(np.sum(np.abs(np.cumsum(p) * dx - q_cdf)) * dx)
    return xs, potential(xs), stat, times, centers, dens, np.asarray(particles), snapshots, np.asarray(kl), np.asarray(wd)


def style_axes(ax, dark=False):
    ax.set_facecolor("#09111f" if dark else "#FBFBFD")
    ax.grid(alpha=0.20, linestyle=":")
    ax.tick_params(colors="#DDE7F3" if dark else "#334155", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#445064" if dark else "#CBD5E1")


def make_picture(data):
    xs, pot, stat, times, centers, dens, _particles, snapshots, kl, wd = data
    fig = plt.figure(figsize=(14.6, 7.1), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[1.2, 1.08, 1.0], hspace=0.34, wspace=0.27)
    ax_a, ax_b, ax_c, ax_d = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    for ax in [ax_a, ax_b, ax_c, ax_d]:
        style_axes(ax)

    pot_norm = (pot - pot.min()) / (pot.max() - pot.min())
    ax_a.fill_between(xs, 0, pot_norm, color="#E9D5FF", alpha=0.35)
    ax_a.plot(xs, pot_norm, color="#111827", lw=2.4, label=r"$f(x)$")
    snap_colors = [COLORS["gf"], COLORS["nag"], COLORS["gd"], "#FF8A00", COLORS["agf"]]
    for t_snap, color in zip(snapshots, snap_colors):
        sample = snapshots[t_snap][::70]
        y = np.interp(sample, xs, pot_norm) + 0.035 * np.sin(np.arange(len(sample)))
        ax_a.scatter(sample, y, s=12, color=color, alpha=0.55)
    ax_a.set_title("particles on $f(x)$", fontsize=11)
    ax_a.set_xlim(xs.min(), xs.max())
    ax_a.set_ylim(-0.05, 1.14)
    ax_a.set_yticks([])

    levels = np.linspace(0, np.percentile(dens, 99.2), 24)
    cf = ax_b.contourf(times, centers, dens.T, levels=levels, cmap="BuPu", extend="max")
    ax_b.contour(times, centers, dens.T, levels=levels[::4], colors="#4B5563", linewidths=0.35, alpha=0.35)
    ax_b.set_title(r"$\rho(x,t)$", fontsize=11)
    ax_b.set_xlabel("time")
    ax_b.set_ylabel("$x$")

    bins = np.r_[centers[0] - (centers[1] - centers[0]) / 2, 0.5 * (centers[1:] + centers[:-1]), centers[-1] + (centers[-1] - centers[-2]) / 2]
    for t_snap, color in zip(snapshots, snap_colors):
        hist, _ = np.histogram(snapshots[t_snap], bins=bins, density=True)
        ax_c.plot(centers, hist, color=color, lw=1.8, label=f"t={t_snap:g}")
    ax_c.plot(xs, stat, color="black", lw=2.1, ls="--", label=r"$\rho^\star$")
    ax_c.set_title(r"$\rho_t$ vs $\rho^\star$", fontsize=11)
    ax_c.set_xlabel("$x$")
    ax_c.legend(fontsize=7.5, ncol=2, framealpha=0.92)

    ax_d.plot(times, kl, color=COLORS["agf"], lw=2.0, label=r"KL$(\rho_t,\rho^\star)$")
    ax_d.plot(times, wd, color=COLORS["gf"], lw=2.0, label=r"$W_1(\rho_t,\rho^\star)$")
    ax_d.set_yscale("log")
    ax_d.set_title("KL / Wasserstein", fontsize=11)
    ax_d.set_xlabel("time")
    ax_d.legend(fontsize=8, framealpha=0.92)
    fig.colorbar(cf, ax=ax_b, fraction=0.046, pad=0.02).set_label("density", fontsize=8)
    fig.suptitle(r"Fokker--Planck: $dX_t=-\nabla f(X_t)dt+\sigma dW_t$", fontsize=13, weight="bold", y=0.985)
    save_pdf(fig, LECTURES / "fokker_planck_evolution.pdf")


def make_video(data):
    xs, pot, stat, times, centers, dens, particles, _snapshots, kl, wd = data
    OPTIM_VIS.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.4, 6.6), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[1.2, 1.08, 1.0], hspace=0.30, wspace=0.25)
    ax_a, ax_b, ax_c, ax_d = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    for ax in [ax_a, ax_b, ax_c, ax_d]:
        style_axes(ax, dark=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.08)
    fig.suptitle("Fokker--Planck", fontsize=13, weight="bold", y=0.98)
    fig.text(0.985, 0.015, "@fminxyz", ha="right", va="bottom", fontsize=8, color="gray", alpha=0.65)

    pot_norm = (pot - pot.min()) / (pot.max() - pot.min())
    ax_a.fill_between(xs, 0, pot_norm, color="#E9D5FF", alpha=0.35)
    ax_a.plot(xs, pot_norm, color="#111827", lw=2.0)
    scat = ax_a.scatter([], [], s=14, color=COLORS["gf"], alpha=0.72)
    time_text = ax_a.text(0.02, 0.86, "", transform=ax_a.transAxes, color="#111827", fontsize=12)
    ax_a.set_xlim(xs.min(), xs.max())
    ax_a.set_ylim(-0.05, 1.13)
    ax_a.set_yticks([])
    ax_a.set_title("particles", fontsize=11)

    levels = np.linspace(0, np.percentile(dens, 99.2), 24)
    ax_b.contourf(times, centers, dens.T, levels=levels, cmap="BuPu", alpha=0.95)
    marker = ax_b.axvline(0, color=COLORS["gf"], lw=3.0)
    ax_b.set_title(r"$\rho(x,t)$", fontsize=11)
    ax_b.set_xlabel("time")
    ax_b.set_ylabel("$x$")

    density_line, = ax_c.plot([], [], color=COLORS["gf"], lw=2.5, label=r"$\rho_t$")
    fill = [None]
    ax_c.plot(xs, stat, color="#111827", lw=1.8, ls="--", label=r"$\rho^\star$")
    ax_c.set_xlim(xs.min(), xs.max())
    ax_c.set_ylim(0, max(dens.max(), stat.max()) * 1.05)
    ax_c.set_title(r"$\rho_t$ vs $\rho^\star$", fontsize=11)
    ax_c.legend(fontsize=8, framealpha=0.4)

    kl_line, = ax_d.plot([], [], color=COLORS["agf"], lw=2.1, label="KL")
    wd_line, = ax_d.plot([], [], color=COLORS["gf"], lw=2.1, label="W1")
    ax_d.set_xlim(times.min(), times.max())
    ax_d.set_ylim(max(min(wd.min(), kl.min()) * 0.65, 1e-3), max(kl.max(), wd.max()) * 1.25)
    ax_d.set_yscale("log")
    ax_d.set_title("KL / W1", fontsize=11)
    ax_d.legend(fontsize=8, framealpha=0.4)

    frames = 160

    def update(i):
        idx = min(len(times) - 1, int(i / (frames - 1) * (len(times) - 1)))
        sample = particles[idx]
        y = np.interp(sample, xs, pot_norm) + 0.018 * np.sin(np.arange(len(sample)) * 1.7 + i * 0.15)
        scat.set_offsets(np.c_[sample, y])
        marker.set_xdata([times[idx], times[idx]])
        density_line.set_data(centers, dens[idx])
        if fill[0] is not None:
            fill[0].remove()
        fill[0] = ax_c.fill_between(centers, 0, dens[idx], color=COLORS["gf"], alpha=0.20)
        kl_line.set_data(times[: idx + 1], kl[: idx + 1])
        wd_line.set_data(times[: idx + 1], wd[: idx + 1])
        time_text.set_text(f"t = {times[idx]:.2f}")
        return [scat, marker, density_line, kl_line, wd_line, time_text, fill[0]]

    out = OPTIM_VIS / "fokker_planck_density.mp4"
    writer = FFMpegWriter(fps=30, codec="h264", bitrate=-1, extra_args=["-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium"])
    FuncAnimation(fig, update, frames=frames, interval=1000 / 30, blit=False).save(out, writer=writer, dpi=190)
    plt.close(fig)
    print(f"saved {out}")


def write_optim_page():
    page = OPTIM_VIS / "fokker_planck_density.md"
    page.write_text(
        """---
title: "Fokker--Planck Density Evolution"
---

:::{.video}
fokker_planck_density.mp4
:::

[Code](fokker_planck_density.py)
""",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), OPTIM_VIS / "fokker_planck_density.py")
    print(f"saved {page}")


def main():
    data = simulate()
    make_picture(data)
    make_video(data)
    write_optim_page()


if __name__ == "__main__":
    main()
