"""
Visualization: L1 Regularization Sparsity Dynamics
Two phases: (1) full convergence, (2) zoom into "zero" components.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from tqdm.auto import tqdm

# ── Parameters ───────────────────────────────────────────
N_ITERS = 1200
PHASE1_FRAMES = 540   # 9s: convergence
PHASE2_FRAMES = 360   # 6s: zoom into zeros
N_FRAMES = PHASE1_FRAMES + PHASE2_FRAMES  # 15s total
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

# ── Optimal x* ───────────────────────────────────────────
x_star = np.zeros(n)
for _ in range(50000):
    x_star = soft_threshold(x_star - (1.0 / L) * grad_f(x_star), lam / L)

# ── Starting point ───────────────────────────────────────
x0 = 0.8 * np.random.randn(n)

# ── Trajectories ─────────────────────────────────────────
traj_sub = [x0.copy()]
x_sub = x0.copy()
x_sub_sum = x0.copy()
G_max = 0.0
R = np.linalg.norm(x0 - x_star)

for k in range(1, N_ITERS + 1):
    g = grad_f(x_sub) + lam * np.sign(x_sub)
    G_max = max(G_max, np.linalg.norm(g))
    alpha_k = R / (max(G_max, 1e-8) * np.sqrt(k))
    x_sub = x_sub - alpha_k * g
    x_sub_sum += x_sub
    traj_sub.append((x_sub_sum / (k + 1)).copy())

traj_sub = np.array(traj_sub)

traj_prox = [x0.copy()]
x_prox = x0.copy()
for _ in range(N_ITERS):
    x_prox = soft_threshold(x_prox - (1.0 / L) * grad_f(x_prox), lam / L)
    traj_prox.append(x_prox.copy())

traj_prox = np.array(traj_prox)

# ── Identify zero components ─────────────────────────────
zero_mask = np.abs(x_star) < 1e-10
zero_idx = np.where(zero_mask)[0]
nonzero_idx = np.where(~zero_mask)[0]

# ── Interpolation ────────────────────────────────────────
def get_state(traj, frame):
    """Smooth state interpolation for phase 1."""
    t = N_ITERS * (frame / (PHASE1_FRAMES - 1)) ** 0.55
    k = min(int(t), N_ITERS - 1)
    frac = t - k
    return (1 - frac) * traj[k] + frac * traj[min(k + 1, N_ITERS)]

def get_iteration(frame):
    t = N_ITERS * (frame / (PHASE1_FRAMES - 1)) ** 0.55
    return min(int(t), N_ITERS)

# ── Colors ───────────────────────────────────────────────
C_SUB = '#E63946'
C_PROX = '#2A9D8F'
C_STAR = '#457B9D'

# ── Figure Setup ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
fig.subplots_adjust(top=0.82, bottom=0.12, left=0.06, right=0.98, wspace=0.25)

idx = np.arange(n)
bar_width = 0.65

ymin = min(traj_sub.min(), traj_prox.min(), x_star.min()) * 1.15
ymax = max(traj_sub.max(), traj_prox.max(), x_star.max()) * 1.15

bars_sub = ax1.bar(idx, x0, bar_width, color=C_SUB, alpha=0.85, zorder=3)
bars_prox = ax2.bar(idx, x0, bar_width, color=C_PROX, alpha=0.85, zorder=3)

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

suptitle = fig.suptitle(r'LASSO: $\ell_1$ sparsity dynamics.  $n=%d$, $\lambda=%.1f$' % (n, lam),
                         fontsize=14, fontweight='bold', y=0.97)

fig.text(0.99, 0.01, '@fminxyz', ha='right', va='bottom', fontsize=8, color='gray', alpha=0.5)

# Final states for phase 2
final_sub = traj_sub[-1]
final_prox = traj_prox[-1]

# Zoom targets for phase 2
zoom_ymin = min(final_sub[zero_idx].min(), -0.01) * 1.3
zoom_ymax = max(final_sub[zero_idx].max(), 0.01) * 1.3
zoom_xmin = zero_idx[0] - 1.0
zoom_xmax = zero_idx[-1] + 1.0

# ── Smooth easing ────────────────────────────────────────
def ease_in_out(t):
    """Smooth ease-in-out from 0 to 1."""
    return t * t * (3 - 2 * t)

# ── Animation ────────────────────────────────────────────
pbar = tqdm(total=100, desc="Rendering")

def update(frame):
    if frame < PHASE1_FRAMES:
        # ── Phase 1: Convergence ──
        state_sub = get_state(traj_sub, frame)
        state_prox = get_state(traj_prox, frame)
        iteration = get_iteration(frame)

        for bar, val in zip(bars_sub, state_sub):
            bar.set_height(val)
        for bar, val in zip(bars_prox, state_prox):
            bar.set_height(val)

        thr = 1e-8
        nz_sub = np.sum(np.abs(state_sub) < thr)
        nz_prox = np.sum(np.abs(state_prox) < thr)

        title_sub.set_text('Subgradient method\nIteration %d | Zeros: %d/%d' % (iteration, nz_sub, n))
        title_prox.set_text('Proximal gradient\nIteration %d | Zeros: %d/%d' % (iteration, nz_prox, n))

    else:
        # ── Phase 2: Zoom into zero components ──
        t = ease_in_out((frame - PHASE1_FRAMES) / (PHASE2_FRAMES - 1))

        # Interpolate axis limits
        cur_ymin = ymin + t * (zoom_ymin - ymin)
        cur_ymax = ymax + t * (zoom_ymax - ymax)
        cur_xmin = -0.5 + t * (zoom_xmin - (-0.5))
        cur_xmax = (n - 0.5) + t * (zoom_xmax - (n - 0.5))

        for ax in [ax1, ax2]:
            ax.set_ylim(cur_ymin, cur_ymax)
            ax.set_xlim(cur_xmin, cur_xmax)

        if t > 0.5:
            title_sub.set_text('Subgradient: components where $x^*_i = 0$\nnoise $\\approx 10^{-3}$, no exact zeros')
            title_prox.set_text('Proximal: components where $x^*_i = 0$\n$x_i = 0$ exactly (soft-thresholding)')

    return list(bars_sub) + list(bars_prox) + [title_sub, title_prox]

ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 // FPS, blit=False)

writer = animation.FFMpegWriter(
    fps=FPS, metadata=dict(artist='@fminxyz'), bitrate=-1,
    codec='h264',
    extra_args=['-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p', '-tune', 'animation']
)

ani.save(FILENAME, writer=writer, dpi=DPI,
         progress_callback=lambda i, n: pbar.update(100 / n))
pbar.close()
plt.close()
print(f"Saved: {FILENAME}")
