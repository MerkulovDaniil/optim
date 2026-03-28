"""
Visualization: L1 Regularization Sparsity Dynamics
Arc-length parameterization: frames distributed by rate of change.
Phase 1: convergence. Phase 2: zoom into zero components.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from tqdm.auto import tqdm

# ── Parameters ───────────────────────────────────────────
N_ITERS = 1500
PHASE1_FRAMES = 540   # 9s convergence
PHASE2_FRAMES = 360   # 6s zoom
N_FRAMES = PHASE1_FRAMES + PHASE2_FRAMES
FPS = 60
DPI = 200
FILENAME = "l1_sparsity.mp4"

np.random.seed(42)

# ── Problem Setup ────────────────────────────────────────
n = 20
m = 150
lam = 1.0
L_H = 8.0

Q, _ = np.linalg.qr(np.random.randn(n, n))
eigs = np.linspace(0.01, L_H, n)
U, _ = np.linalg.qr(np.random.randn(m, n), mode='reduced')
A = U @ np.diag(np.sqrt(eigs * m)) @ Q.T

w_true = np.zeros(n)
w_true[:7] = np.array([3.0, -2.5, 1.8, -1.2, 2.0, -0.8, 1.5])
b = A @ w_true + 0.05 * np.random.randn(m)
L = L_H

def grad_f(x):
    return (1.0 / m) * A.T @ (A @ x - b)

def soft_threshold(x, kappa):
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)

x_star = np.zeros(n)
for _ in range(50000):
    x_star = soft_threshold(x_star - (1.0 / L) * grad_f(x_star), lam / L)

x0 = 0.8 * np.random.randn(n)

# ── Trajectories ─────────────────────────────────────────
traj_sub = [x0.copy()]
x_sub = x0.copy()
x_sub_sum = x0.copy()
G_max, R = 0.0, np.linalg.norm(x0 - x_star)

for k in range(1, N_ITERS + 1):
    g = grad_f(x_sub) + lam * np.sign(x_sub)
    G_max = max(G_max, np.linalg.norm(g))
    x_sub = x_sub - R / (max(G_max, 1e-8) * np.sqrt(k)) * g
    x_sub_sum += x_sub
    traj_sub.append((x_sub_sum / (k + 1)).copy())
traj_sub = np.array(traj_sub)

traj_prox = [x0.copy()]
x_prox = x0.copy()
for _ in range(N_ITERS):
    x_prox = soft_threshold(x_prox - (1.0 / L) * grad_f(x_prox), lam / L)
    traj_prox.append(x_prox.copy())
traj_prox = np.array(traj_prox)

# ── Arc-length parameterization ──────────────────────────
# Measure combined change across both methods per iteration
deltas = np.array([
    np.linalg.norm(traj_sub[k+1] - traj_sub[k]) +
    np.linalg.norm(traj_prox[k+1] - traj_prox[k])
    for k in range(N_ITERS)
])
# Cumulative arc length
cumlen = np.concatenate([[0], np.cumsum(deltas)])
cumlen_norm = cumlen / cumlen[-1]  # normalize to [0, 1]

# Map each phase-1 frame to an iteration via arc-length
def frame_to_iter_arclength(frame_idx):
    """Map frame [0..PHASE1_FRAMES-1] -> iteration [0..N_ITERS] via arc length."""
    t = frame_idx / (PHASE1_FRAMES - 1)  # uniform in [0, 1]
    # Find iteration where cumlen_norm >= t
    return int(np.searchsorted(cumlen_norm, t))

# Precompute mapping
phase1_iters = [frame_to_iter_arclength(f) for f in range(PHASE1_FRAMES)]

# Verify: print distribution
bins = [0, 10, 50, 100, 500, 1000, N_ITERS]
for i in range(len(bins) - 1):
    count = sum(1 for it in phase1_iters if bins[i] <= it < bins[i+1])
    print(f"  Iterations {bins[i]:4d}-{bins[i+1]:4d}: {count:3d} frames")

# Smooth interpolation between iterations
def get_state(traj, frame):
    if frame >= PHASE1_FRAMES:
        return traj[-1]
    t_norm = frame / (PHASE1_FRAMES - 1)
    # Exact arc-length position
    arc_pos = np.searchsorted(cumlen_norm, t_norm, side='right') - 1
    arc_pos = max(0, min(arc_pos, N_ITERS - 1))
    # Fractional interpolation within the iteration
    if cumlen_norm[arc_pos + 1] > cumlen_norm[arc_pos]:
        frac = (t_norm - cumlen_norm[arc_pos]) / (cumlen_norm[arc_pos + 1] - cumlen_norm[arc_pos])
    else:
        frac = 0
    frac = max(0, min(frac, 1))
    return (1 - frac) * traj[arc_pos] + frac * traj[arc_pos + 1]

# ── Identify zero components ─────────────────────────────
zero_mask = np.abs(x_star) < 1e-10
zero_idx = np.where(zero_mask)[0]

# ── Colors ───────────────────────────────────────────────
C_SUB = '#E63946'
C_PROX = '#2A9D8F'
C_STAR = '#457B9D'

# ── Figure ───────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
fig.subplots_adjust(top=0.82, bottom=0.12, left=0.06, right=0.98, wspace=0.25)

idx = np.arange(n)
bw = 0.65

ymin = min(traj_sub.min(), traj_prox.min(), x_star.min()) * 1.15
ymax = max(traj_sub.max(), traj_prox.max(), x_star.max()) * 1.15

bars_sub = ax1.bar(idx, x0, bw, color=C_SUB, alpha=0.85, zorder=3)
bars_prox = ax2.bar(idx, x0, bw, color=C_PROX, alpha=0.85, zorder=3)

for ax in [ax1, ax2]:
    ax.scatter(idx, x_star, s=25, color=C_STAR, zorder=5, marker='o',
               edgecolors='white', linewidths=0.4, label=r'$x^*$')
    ax.axhline(0, color='black', lw=0.5, zorder=2)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel('Component $i$', fontsize=11)
    ax.set_ylabel('$x_i$', fontsize=12)
    ax.grid(True, alpha=0.15, axis='y')
    ax.legend(fontsize=9, loc='lower right')

title_sub = ax1.set_title('Subgradient method\nIteration 0 | Zeros: 0/%d' % n, fontsize=11)
title_prox = ax2.set_title('Proximal gradient\nIteration 0 | Zeros: 0/%d' % n, fontsize=11)
fig.suptitle(r'LASSO: $\ell_1$ sparsity dynamics.  $n=%d$, $\lambda=%.1f$' % (n, lam),
             fontsize=14, fontweight='bold', y=0.97)
fig.text(0.99, 0.01, '@fminxyz', ha='right', va='bottom', fontsize=8, color='gray', alpha=0.5)

# Zoom targets
final_sub = traj_sub[-1]
zoom_ymin = min(final_sub[zero_idx].min(), -0.01) * 1.5
zoom_ymax = max(final_sub[zero_idx].max(), 0.01) * 1.5
zoom_xmin = zero_idx[0] - 1.0
zoom_xmax = zero_idx[-1] + 1.0

def ease(t):
    return t * t * (3 - 2 * t)

# ── Animation ────────────────────────────────────────────
pbar = tqdm(total=100, desc="Rendering")

def update(frame):
    if frame < PHASE1_FRAMES:
        state_sub = get_state(traj_sub, frame)
        state_prox = get_state(traj_prox, frame)
        iteration = phase1_iters[frame]

        for bar, val in zip(bars_sub, state_sub):
            bar.set_height(val)
        for bar, val in zip(bars_prox, state_prox):
            bar.set_height(val)

        nz_sub = np.sum(np.abs(state_sub) < 1e-8)
        nz_prox = np.sum(np.abs(state_prox) < 1e-8)
        title_sub.set_text('Subgradient method\nIteration %d | Zeros: %d/%d' % (iteration, nz_sub, n))
        title_prox.set_text('Proximal gradient\nIteration %d | Zeros: %d/%d' % (iteration, nz_prox, n))
    else:
        t = ease((frame - PHASE1_FRAMES) / (PHASE2_FRAMES - 1))

        cur_ymin = ymin + t * (zoom_ymin - ymin)
        cur_ymax = ymax + t * (zoom_ymax - ymax)
        cur_xmin = -0.5 + t * (zoom_xmin + 0.5)
        cur_xmax = (n - 0.5) + t * (zoom_xmax - n + 0.5)

        for ax in [ax1, ax2]:
            ax.set_ylim(cur_ymin, cur_ymax)
            ax.set_xlim(cur_xmin, cur_xmax)

        if t > 0.5:
            title_sub.set_text('Subgradient: $x^*_i = 0$ components\nnoise $\\approx 10^{-3}$, no exact zeros')
            title_prox.set_text('Proximal: $x^*_i = 0$ components\n$x_i = 0$ exactly (soft-thresholding)')

    return list(bars_sub) + list(bars_prox) + [title_sub, title_prox]

ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 // FPS, blit=False)

writer = animation.FFMpegWriter(
    fps=FPS, metadata=dict(artist='@fminxyz'), bitrate=-1, codec='h264',
    extra_args=['-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p', '-tune', 'animation']
)
ani.save(FILENAME, writer=writer, dpi=DPI,
         progress_callback=lambda i, n: pbar.update(100 / n))
pbar.close()
plt.close()
print(f"Saved: {FILENAME}")
