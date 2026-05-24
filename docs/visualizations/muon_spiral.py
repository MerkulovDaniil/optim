"""
SGD vs Nesterov vs AdamW vs Muon: 2D spiral classification, mini-batch training.
Muon for W1 (matrix), AdamW for other params (as in practice).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import ListedColormap

matplotlib.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})
COLORS = {'SGD': '#4C72B0', 'Nesterov': '#DD8452', 'AdamW': '#55A868', 'Muon': '#8172B3'}
cmap_bg = ListedColormap(['#dce6f5', '#f5dce0'])

# ── Data ──
np.random.seed(42)
n_pts = 400; n_half = n_pts // 2; batch_size = 40
theta = np.linspace(0.5, 3.2 * np.pi, n_half)
r = np.linspace(0.3, 2.0, n_half)
noise = 0.15
x1 = np.stack([r*np.cos(theta)+np.random.randn(n_half)*noise,
               r*np.sin(theta)+np.random.randn(n_half)*noise], axis=1)
x2 = np.stack([r*np.cos(theta+np.pi)+np.random.randn(n_half)*noise,
               r*np.sin(theta+np.pi)+np.random.randn(n_half)*noise], axis=1)
X_all = np.vstack([x1, x2])
y_all = np.concatenate([np.zeros(n_half), np.ones(n_half)])
X_aug = np.hstack([X_all, np.ones((n_pts,1))]).T   # (3, n)

grid_res = 100
xx, yy = np.meshgrid(np.linspace(-2.8, 2.8, grid_res), np.linspace(-2.8, 2.8, grid_res))
X_grid = np.c_[xx.ravel(), yy.ravel(), np.ones(grid_res**2)].T

# ── Network: z=tanh(W1@x), y_hat=sigmoid(W2@z) ──
h = 64

def sigmoid(x):
    return np.where(x>=0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

def forward(W1, W2, X):
    Z = np.tanh(W1 @ X)
    logits = (W2 @ Z).ravel()
    return Z, logits, sigmoid(logits)

def bce_loss(y_hat, y):
    y_hat = np.clip(y_hat, 1e-7, 1-1e-7)
    return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

def grads(W1, W2, X, y):
    n = X.shape[1]
    Z, _, y_hat = forward(W1, W2, X)
    delta = (y_hat - y) / n
    dW2 = delta[None,:] @ Z.T
    dW1 = ((1-Z**2) * (W2.T @ delta[None,:])) @ X.T
    return dW1, dW2

def predict_grid(W1, W2):
    _, _, y_hat = forward(W1, W2, X_grid)
    return y_hat.reshape(grid_res, grid_res)

def accuracy(W1, W2):
    _, _, yh = forward(W1, W2, X_aug)
    return np.mean((yh > 0.5) == y_all)

def full_loss(W1, W2):
    _, _, yh = forward(W1, W2, X_aug)
    return bce_loss(yh, y_all)

# ── Newton-Schulz (handles non-square) ──
def newton_schulz(G, steps=7):
    a, b, c = 3.4445, -4.7750, 2.0315
    nrm = np.linalg.norm(G, 'fro')
    if nrm < 1e-15:
        return np.zeros_like(G)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        G = G.T
    X = G / nrm
    for _ in range(steps):
        A_ = X @ X.T
        X = a*X + b*A_@X + c*(A_@A_)@X
    return X.T if transposed else X

# ── Init ──
np.random.seed(7)
W1_init = np.random.randn(h, 3) * 0.15
W2_init = np.random.randn(1, h) * 0.15

def get_batch():
    idx = np.random.choice(n_pts, batch_size, replace=False)
    return X_aug[:, idx], y_all[idx]

# ── Optimizers (mini-batch) ──
n_steps = 3000
save_every = 1

def run_sgd(lr, wd=0):
    W1, W2 = W1_init.copy(), W2_init.copy()
    losses, preds = [], []
    for i in range(n_steps):
        if i % save_every == 0:
            losses.append(full_loss(W1, W2))
            preds.append(predict_grid(W1, W2))
        Xb, yb = get_batch()
        dW1, dW2 = grads(W1, W2, Xb, yb)
        W1 = W1*(1-lr*wd) - lr*dW1
        W2 = W2*(1-lr*wd) - lr*dW2
    losses.append(full_loss(W1, W2)); preds.append(predict_grid(W1, W2))
    return losses, preds

def run_nesterov(lr, mu, wd=0):
    W1, W2 = W1_init.copy(), W2_init.copy()
    V1, V2 = np.zeros_like(W1), np.zeros_like(W2)
    losses, preds = [], []
    for i in range(n_steps):
        if i % save_every == 0:
            losses.append(full_loss(W1, W2))
            preds.append(predict_grid(W1, W2))
        Xb, yb = get_batch()
        dW1, dW2 = grads(W1+mu*V1, W2+mu*V2, Xb, yb)
        V1 = mu*V1 - lr*dW1; V2 = mu*V2 - lr*dW2
        W1 = W1*(1-lr*wd)+V1; W2 = W2*(1-lr*wd)+V2
    losses.append(full_loss(W1, W2)); preds.append(predict_grid(W1, W2))
    return losses, preds

def run_adamw(lr, b1, b2, wd):
    W1, W2 = W1_init.copy(), W2_init.copy()
    m1,v1 = np.zeros_like(W1), np.zeros_like(W1)
    m2,v2 = np.zeros_like(W2), np.zeros_like(W2)
    losses, preds = [], []
    for i in range(n_steps):
        t = i+1
        if i % save_every == 0:
            losses.append(full_loss(W1, W2))
            preds.append(predict_grid(W1, W2))
        Xb, yb = get_batch()
        dW1, dW2 = grads(W1, W2, Xb, yb)
        for W, dW, m, v in [(W1,dW1,m1,v1),(W2,dW2,m2,v2)]:
            m[:] = b1*m + (1-b1)*dW
            v[:] = b2*v + (1-b2)*dW**2
            mh = m/(1-b1**t); vh = v/(1-b2**t)
            W[:] = W*(1-lr*wd) - lr*mh/(np.sqrt(vh)+1e-8)
    losses.append(full_loss(W1, W2)); preds.append(predict_grid(W1, W2))
    return losses, preds

def run_muon(lr_muon, lr_adam, mu_muon, b1, b2, wd):
    W1, W2 = W1_init.copy(), W2_init.copy()
    Buf1 = np.zeros_like(W1)
    m2,v2 = np.zeros_like(W2), np.zeros_like(W2)
    losses, preds = [], []
    for i in range(n_steps):
        t = i+1
        if i % save_every == 0:
            losses.append(full_loss(W1, W2))
            preds.append(predict_grid(W1, W2))
        Xb, yb = get_batch()
        dW1, dW2 = grads(W1, W2, Xb, yb)
        Buf1 = mu_muon*Buf1 + dW1
        G_n = mu_muon*Buf1 + dW1
        O = newton_schulz(G_n)
        W1 = W1*(1-lr_muon*wd) - lr_muon*O
        m2[:] = b1*m2 + (1-b1)*dW2
        v2[:] = b2*v2 + (1-b2)*dW2**2
        mh = m2/(1-b1**t); vh = v2/(1-b2**t)
        W2[:] = W2*(1-lr_adam*wd) - lr_adam*mh/(np.sqrt(vh)+1e-8)
    losses.append(full_loss(W1, W2)); preds.append(predict_grid(W1, W2))
    return losses, preds

# ── Run with tuned hyperparams ──
np.random.seed(0)
loss_sgd, preds_sgd = run_sgd(lr=0.23, wd=1e-4)
np.random.seed(0)
loss_nest, preds_nest = run_nesterov(lr=1.4, mu=0.9, wd=1e-4)
np.random.seed(0)
loss_adam, preds_adam = run_adamw(lr=0.37, b1=0.9, b2=0.999, wd=1e-4)
np.random.seed(0)
loss_muon, preds_muon = run_muon(lr_muon=0.44, lr_adam=0.37, mu_muon=0.95, b1=0.9, b2=0.999, wd=1e-4)

for name, ls in [('SGD',loss_sgd),('Nest',loss_nest),('Adam',loss_adam),('Muon',loss_muon)]:
    print(f"{name}: final_loss={ls[-1]:.4f}")

# ── Static figure ──
snap_steps = [0, 200, 600, 1500, 3000]
snap_indices = [s // save_every for s in snap_steps]

methods = [
    ('SGD',      preds_sgd,  loss_sgd),
    ('Nesterov', preds_nest, loss_nest),
    ('AdamW',    preds_adam, loss_adam),
    ('Muon',     preds_muon, loss_muon),
]

n_snaps = len(snap_steps)

# ── File 1: decision boundary snapshots ──
fig1 = plt.figure(figsize=(2.6*n_snaps, 2.6*4 + 0.6))
gs1 = GridSpec(4, n_snaps, wspace=0.06, hspace=0.22)

for row, (name, preds, _) in enumerate(methods):
    for col, (si, st) in enumerate(zip(snap_indices, snap_steps)):
        ax = fig1.add_subplot(gs1[row, col])
        si_c = min(si, len(preds)-1)
        ax.contourf(xx, yy, preds[si_c], levels=[0,0.5,1], cmap=cmap_bg, alpha=0.85)
        ax.contour(xx, yy, preds[si_c], levels=[0.5], colors='k', linewidths=1.5, alpha=0.7)
        ax.scatter(X_all[:n_half,0], X_all[:n_half,1], c='#2060b0', s=5, alpha=0.5, edgecolors='none')
        ax.scatter(X_all[n_half:,0], X_all[n_half:,1], c='#c03040', s=5, alpha=0.5, edgecolors='none')
        ax.set_xlim(-2.8,2.8); ax.set_ylim(-2.8,2.8); ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
        if row == 0: ax.set_title(f'$k={st}$', fontsize=11)
        if col == 0:
            ax.set_ylabel(name, fontsize=13, fontweight='bold',
                          color=COLORS[name], rotation=0, labelpad=55, va='center')

fig1.suptitle(r'Классификация спиралей (mini-batch): SGD, Nesterov, AdamW, Muon', fontsize=13, y=0.995)
fig1.savefig('/root/hse26_repo/files/exp_optimizer_boundaries.pdf', bbox_inches='tight', dpi=200)
fig1.savefig('/tmp/exp_optimizer_boundaries.png', bbox_inches='tight', dpi=150)
print("Saved boundaries")
plt.close(fig1)

# ── File 2: loss curves (raw + EMA) ──
def ema(arr, alpha=0.92):
    s = np.empty_like(arr, dtype=float)
    s[0] = arr[0]
    for i in range(1, len(arr)):
        s[i] = alpha * s[i-1] + (1 - alpha) * arr[i]
    return s

fig2, ax_loss = plt.subplots(figsize=(7, 4.5))
loss_x = np.linspace(0, n_steps, len(loss_sgd))
for name, _, losses in methods:
    raw = np.array(losses)
    ax_loss.semilogy(loss_x, raw, color=COLORS[name], lw=0.5, alpha=0.2)
    ax_loss.semilogy(loss_x, ema(raw), color=COLORS[name], lw=2.5, label=name)
ax_loss.set_xlabel('Шаг')
ax_loss.set_ylabel('BCE Loss')
ax_loss.set_title(r'Сходимость: 1-hidden-layer NN на спиралях (mini-batch)')
ax_loss.legend(frameon=True, fancybox=False, edgecolor='#ccc', fontsize=11)
ax_loss.grid(True, alpha=0.3)
ax_loss.set_xlim(0, n_steps)
ax_loss.spines['top'].set_visible(False)
ax_loss.spines['right'].set_visible(False)
fig2.savefig('/root/hse26_repo/files/exp_optimizer_loss.pdf', bbox_inches='tight', dpi=200)
fig2.savefig('/tmp/exp_optimizer_loss.png', bbox_inches='tight', dpi=150)
print("Saved loss curves")
plt.close(fig2)

# ── Animation ──
fig_a, axes_a = plt.subplots(1, 4, figsize=(12, 3.2))
fig_a.subplots_adjust(wspace=0.05, top=0.85)
title_a = fig_a.suptitle('$k = 0$', fontsize=14)

for i, (name, _, _) in enumerate(methods):
    axes_a[i].set_xlim(-2.8,2.8); axes_a[i].set_ylim(-2.8,2.8)
    axes_a[i].set_xticks([]); axes_a[i].set_yticks([]); axes_a[i].set_aspect('equal')
    axes_a[i].set_title(name, fontsize=13, fontweight='bold', color=COLORS[name])

n_frames = min(len(p) for _, p, _ in methods)

def update_anim(frame):
    for i, (name, preds, _) in enumerate(methods):
        ax = axes_a[i]
        for coll in list(ax.collections):
            coll.remove()
        f = min(frame, len(preds)-1)
        ax.contourf(xx, yy, preds[f], levels=[0,0.5,1], cmap=cmap_bg, alpha=0.85)
        ax.contour(xx, yy, preds[f], levels=[0.5], colors='k', linewidths=1.5, alpha=0.7)
        ax.scatter(X_all[:n_half,0], X_all[:n_half,1], c='#2060b0', s=8, alpha=0.5, edgecolors='none')
        ax.scatter(X_all[n_half:,0], X_all[n_half:,1], c='#c03040', s=8, alpha=0.5, edgecolors='none')
    title_a.set_text(f'$k = {frame}$')

ani = FuncAnimation(fig_a, update_anim, frames=range(n_frames), blit=False, interval=16)
writer = FFMpegWriter(fps=60, bitrate=8000, codec='libx264',
                      extra_args=['-pix_fmt', 'yuv420p', '-preset', 'slow', '-crf', '18'])
ani.save('/tmp/exp_optimizer_raw.mp4', writer=writer)
print("Saved animation")
plt.close()
