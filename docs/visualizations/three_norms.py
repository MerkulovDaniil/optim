"""
SGD vs SignGD vs Muon: steepest descent in three norms.
Frobenius (SGD), ℓ∞ (SignGD), Spectral (Muon).
Shows update directions and convergence on ill-conditioned matrix quadratic.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
import subprocess

matplotlib.rcParams.update({
    'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})

COLORS = {'SGD': '#4C72B0', 'SignGD': '#DD8452', 'Muon': '#8172B3'}

np.random.seed(42)
d = 8

# Ill-conditioned quadratic: f(W) = 0.5 * ||Σ W||_F^2
sigma = np.exp(np.linspace(0, np.log(20), d))
Sigma2 = np.diag(sigma**2)

W0 = np.random.randn(d, d) * 0.5

def loss(W):
    return 0.5 * np.sum((np.diag(sigma) @ W)**2)

def grad(W):
    return Sigma2 @ W

def newton_schulz(G, steps=7):
    a, b, c = 3.4445, -4.7750, 2.0315
    nrm = np.linalg.norm(G, 'fro')
    if nrm < 1e-15: return np.zeros_like(G)
    X = G / nrm
    for _ in range(steps):
        A_ = X @ X.T; X = a*X + b*A_@X + c*(A_@A_)@X
    return X

N = 400

# Run all three
def safe_svd(M):
    try:
        return np.linalg.svd(M, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.zeros(min(M.shape))

def run_sgd(lr, n):
    W = W0.copy(); losses = [loss(W)]; sv_updates = []
    for _ in range(n):
        G = grad(W)
        nrm = np.linalg.norm(G, 'fro')
        D = -G / (nrm + 1e-15)
        sv_updates.append(safe_svd(D))
        W = W - lr * G
        l = loss(W)
        if not np.isfinite(l): break
        losses.append(l)
    return losses, sv_updates

def run_signgd(lr, n):
    W = W0.copy(); losses = [loss(W)]; sv_updates = []
    for _ in range(n):
        G = grad(W)
        D = -np.sign(G)
        sv_updates.append(safe_svd(D))
        W = W - lr * np.sign(G)
        l = loss(W)
        if not np.isfinite(l): break
        losses.append(l)
    return losses, sv_updates

def run_muon(lr, n):
    W = W0.copy(); losses = [loss(W)]; sv_updates = []
    for _ in range(n):
        G = grad(W)
        O = newton_schulz(G)
        sv_updates.append(safe_svd(O))
        W = W - lr * O
        l = loss(W)
        if not np.isfinite(l): break
        losses.append(l)
    return losses, sv_updates

# Tune LRs
def best_lr(run_fn, candidates):
    best_v, best_l = np.inf, candidates[0]
    for lr in candidates:
        ls, _ = run_fn(lr, N)
        fv = ls[-1]
        if np.isfinite(fv) and fv < best_v:
            best_v, best_l = fv, lr
    return best_l

lr_sgd = best_lr(run_sgd, np.logspace(-5, -1, 30))
lr_sign = best_lr(run_signgd, np.logspace(-4, 0, 30))
lr_muon = best_lr(run_muon, np.logspace(-3, 0, 30))
print(f"LRs: SGD={lr_sgd:.5f} SignGD={lr_sign:.5f} Muon={lr_muon:.5f}")

loss_sgd, sv_sgd = run_sgd(lr_sgd, N)
loss_sign, sv_sign = run_signgd(lr_sign, N)
loss_muon, sv_muon = run_muon(lr_muon, N)

L0 = loss_sgd[0]
methods = [('SGD', loss_sgd, sv_sgd), ('SignGD', loss_sign, sv_sign), ('Muon', loss_muon, sv_muon)]

# ── Static figure ──
fig = plt.figure(figsize=(13, 4.5))
gs = GridSpec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.3)

# Left: convergence
ax1 = fig.add_subplot(gs[0])
for name, ls, _ in methods:
    vals = np.array(ls) / L0; vals = np.maximum(vals, 1e-16)
    ax1.semilogy(vals, color=COLORS[name], lw=2.2, label=name)
ax1.set_xlabel('Итерация'); ax1.set_ylabel('$f(W_k) / f(W_0)$')
ax1.set_title('Сходимость')
ax1.legend(frameon=True, edgecolor='#ccc'); ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, N)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# Middle: SV spectrum of update at iter 0
ax2 = fig.add_subplot(gs[1])
x = np.arange(d); w = 0.25
for i, (name, _, svs) in enumerate(methods):
    sv0 = svs[0]; sv0_n = sv0 / (sv0[0] + 1e-15)
    ax2.bar(x + i*w, sv0_n, width=w, color=COLORS[name], alpha=0.8, label=name)
ax2.set_xlabel('Индекс $i$'); ax2.set_ylabel('$\\sigma_i / \\sigma_1$')
ax2.set_title('Спектр обновления (iter 0)')
ax2.set_xticks(x + w); ax2.set_xticklabels(range(1, d+1))
ax2.legend(frameon=True, edgecolor='#ccc', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

# Right: SV spectrum at iter 50
ax3 = fig.add_subplot(gs[2])
it_snap = 50
for i, (name, _, svs) in enumerate(methods):
    sv_s = svs[min(it_snap, len(svs)-1)]
    sv_n = sv_s / (sv_s[0] + 1e-15)
    ax3.bar(x + i*w, sv_n, width=w, color=COLORS[name], alpha=0.8, label=name)
ax3.set_xlabel('Индекс $i$'); ax3.set_ylabel('$\\sigma_i / \\sigma_1$')
ax3.set_title(f'Спектр обновления (iter {it_snap})')
ax3.set_xticks(x + w); ax3.set_xticklabels(range(1, d+1))
ax3.legend(frameon=True, edgecolor='#ccc', fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

fig.suptitle(r'Наискорейший спуск в трёх нормах: $\|\cdot\|_F$ (SGD), $\|\cdot\|_\infty$ (SignGD), $\|\cdot\|_\sigma$ (Muon)',
             fontsize=13, y=0.99)

fig.savefig('/root/hse26_repo/files/exp_sgd_signgd_muon.pdf', bbox_inches='tight', dpi=200)
fig.savefig('/tmp/exp_sgd_signgd_muon.png', bbox_inches='tight', dpi=150)
print("Saved static")
plt.close()

# ── Animation: SV bar chart race ──
fig_a = plt.figure(figsize=(12, 4))
gs_a = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.25)

ax_loss_a = fig_a.add_subplot(gs_a[0])
ax_bars_a = fig_a.add_subplot(gs_a[1])

# Loss lines (grow over time)
lines = {}
for name, _, _ in methods:
    line, = ax_loss_a.semilogy([], [], color=COLORS[name], lw=2.5, label=name)
    lines[name] = line
ax_loss_a.set_xlim(0, N); ax_loss_a.set_ylim(1e-14, 2)
ax_loss_a.set_xlabel('Итерация'); ax_loss_a.set_ylabel('$f(W_k) / f(W_0)$')
ax_loss_a.set_title('Сходимость'); ax_loss_a.legend(frameon=True, edgecolor='#ccc')
ax_loss_a.grid(True, alpha=0.3)
ax_loss_a.spines['top'].set_visible(False); ax_loss_a.spines['right'].set_visible(False)

# SV bars
bar_groups = {}
x = np.arange(d); w = 0.25
for i, (name, _, _) in enumerate(methods):
    bar_groups[name] = ax_bars_a.bar(x + i*w, np.zeros(d), width=w, color=COLORS[name], alpha=0.8, label=name)
ax_bars_a.set_ylim(0, 1.15)
ax_bars_a.set_xlabel('Индекс $i$'); ax_bars_a.set_ylabel('$\\sigma_i / \\sigma_1$')
bars_title = ax_bars_a.set_title('Спектр обновления')
ax_bars_a.set_xticks(x + w); ax_bars_a.set_xticklabels(range(1, d+1))
ax_bars_a.legend(frameon=True, edgecolor='#ccc', fontsize=9)
ax_bars_a.grid(True, alpha=0.3, axis='y')
ax_bars_a.spines['top'].set_visible(False); ax_bars_a.spines['right'].set_visible(False)

fig_a.suptitle(r'SGD ($\|\cdot\|_F$) vs SignGD ($\|\cdot\|_\infty$) vs Muon ($\|\cdot\|_\sigma$)', fontsize=13)
fig_a.tight_layout(rect=[0, 0, 1, 0.92])

def update_anim(frame):
    for name, ls, svs in methods:
        vals = np.array(ls[:frame+1]) / L0
        vals = np.maximum(vals, 1e-16)
        lines[name].set_data(range(len(vals)), vals)
        if frame < len(svs):
            sv = svs[frame]; sv_n = sv / (sv[0] + 1e-15)
            for bar, val in zip(bar_groups[name], sv_n):
                bar.set_height(val)
    bars_title.set_text(f'Спектр обновления (iter {frame})')

ani = FuncAnimation(fig_a, update_anim, frames=range(0, N, 1), blit=False, interval=16)
writer = FFMpegWriter(fps=60, bitrate=5000, codec='libx264',
                      extra_args=['-pix_fmt', 'yuv420p', '-preset', 'slow', '-crf', '18'])
ani.save('/tmp/sgd_signgd_muon_raw.mp4', writer=writer)
plt.close()

dur = N / 60
subprocess.run(['ffmpeg', '-y', '-i', '/tmp/sgd_signgd_muon_raw.mp4',
                '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=mono', '-t', str(dur),
                '-c:v', 'copy', '-c:a', 'aac', '-shortest',
                '/root/hse26_repo/files/exp_sgd_signgd_muon.mp4'],
               capture_output=True)
print(f"Saved animation ({dur:.1f}s)")
