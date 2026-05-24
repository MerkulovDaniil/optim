"""
Hyperparameter tuning with fixed compute budget for SGD, Nesterov, AdamW, Muon.
Same spiral classification setup. Budget: 50 random HP configs × 1000 steps each.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from itertools import product
import json

matplotlib.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})
COLORS = {'SGD': '#4C72B0', 'Nesterov': '#DD8452', 'AdamW': '#55A868', 'Muon': '#8172B3'}
cmap_bg = ListedColormap(['#dce6f5', '#f5dce0'])

# ── Data (same as main experiment) ──
np.random.seed(42)
n_pts = 400; n_half = n_pts // 2; batch_size = 40
theta = np.linspace(0.5, 3.2 * np.pi, n_half)
r = np.linspace(0.3, 2.0, n_half); noise = 0.15
x1 = np.stack([r*np.cos(theta)+np.random.randn(n_half)*noise,
               r*np.sin(theta)+np.random.randn(n_half)*noise], axis=1)
x2 = np.stack([r*np.cos(theta+np.pi)+np.random.randn(n_half)*noise,
               r*np.sin(theta+np.pi)+np.random.randn(n_half)*noise], axis=1)
X_all = np.vstack([x1, x2])
y_all = np.concatenate([np.zeros(n_half), np.ones(n_half)])
X_aug = np.hstack([X_all, np.ones((n_pts,1))]).T

grid_res = 100
xx, yy = np.meshgrid(np.linspace(-2.8, 2.8, grid_res), np.linspace(-2.8, 2.8, grid_res))
X_grid = np.c_[xx.ravel(), yy.ravel(), np.ones(grid_res**2)].T

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

def full_loss(W1, W2):
    _, _, yh = forward(W1, W2, X_aug)
    return bce_loss(yh, y_all)

def accuracy(W1, W2):
    _, _, yh = forward(W1, W2, X_aug)
    return np.mean((yh > 0.5) == y_all)

def newton_schulz(G, steps=7):
    a, b, c = 3.4445, -4.7750, 2.0315
    nrm = np.linalg.norm(G, 'fro')
    if nrm < 1e-15: return np.zeros_like(G)
    transposed = G.shape[0] > G.shape[1]
    if transposed: G = G.T
    X = G / nrm
    for _ in range(steps):
        A_ = X @ X.T
        X = a*X + b*A_@X + c*(A_@A_)@X
    return X.T if transposed else X

np.random.seed(7)
W1_init = np.random.randn(h, 3) * 0.15
W2_init = np.random.randn(1, h) * 0.15

def get_batch(rng):
    idx = rng.choice(n_pts, batch_size, replace=False)
    return X_aug[:, idx], y_all[idx]

N_TUNE = 1000  # steps per trial

def train_sgd(lr, wd, seed=0):
    W1, W2 = W1_init.copy(), W2_init.copy()
    rng = np.random.RandomState(seed)
    for i in range(N_TUNE):
        Xb, yb = get_batch(rng)
        dW1, dW2 = grads(W1, W2, Xb, yb)
        W1 = W1*(1-lr*wd) - lr*dW1
        W2 = W2*(1-lr*wd) - lr*dW2
    return full_loss(W1, W2), W1, W2

def train_nesterov(lr, mu, wd, seed=0):
    W1, W2 = W1_init.copy(), W2_init.copy()
    V1, V2 = np.zeros_like(W1), np.zeros_like(W2)
    rng = np.random.RandomState(seed)
    for i in range(N_TUNE):
        Xb, yb = get_batch(rng)
        dW1, dW2 = grads(W1+mu*V1, W2+mu*V2, Xb, yb)
        V1 = mu*V1 - lr*dW1; V2 = mu*V2 - lr*dW2
        W1 = W1*(1-lr*wd)+V1; W2 = W2*(1-lr*wd)+V2
    return full_loss(W1, W2), W1, W2

def train_adamw(lr, b1, b2, wd, seed=0):
    W1, W2 = W1_init.copy(), W2_init.copy()
    m1,v1 = np.zeros_like(W1), np.zeros_like(W1)
    m2,v2 = np.zeros_like(W2), np.zeros_like(W2)
    rng = np.random.RandomState(seed)
    for i in range(N_TUNE):
        t = i+1
        Xb, yb = get_batch(rng)
        dW1, dW2 = grads(W1, W2, Xb, yb)
        for W, dW, m, v in [(W1,dW1,m1,v1),(W2,dW2,m2,v2)]:
            m[:] = b1*m + (1-b1)*dW
            v[:] = b2*v + (1-b2)*dW**2
            mh = m/(1-b1**t); vh = v/(1-b2**t)
            W[:] = W*(1-lr*wd) - lr*mh/(np.sqrt(vh)+1e-8)
    return full_loss(W1, W2), W1, W2

def train_muon(lr_muon, lr_adam, mu_muon, b1, b2, wd, seed=0):
    W1, W2 = W1_init.copy(), W2_init.copy()
    Buf1 = np.zeros_like(W1)
    m2,v2 = np.zeros_like(W2), np.zeros_like(W2)
    rng = np.random.RandomState(seed)
    for i in range(N_TUNE):
        t = i+1
        Xb, yb = get_batch(rng)
        dW1, dW2 = grads(W1, W2, Xb, yb)
        Buf1 = mu_muon*Buf1 + dW1
        G_n = mu_muon*Buf1 + dW1
        O = newton_schulz(G_n)
        W1 = W1*(1-lr_muon*wd) - lr_muon*O
        m2[:] = b1*m2 + (1-b1)*dW2
        v2[:] = b2*v2 + (1-b2)*dW2**2
        mh = m2/(1-b1**t); vh = v2/(1-b2**t)
        W2[:] = W2*(1-lr_adam*wd) - lr_adam*mh/(np.sqrt(vh)+1e-8)
    return full_loss(W1, W2), W1, W2

# ── HP search grids ──
print("=== Hyperparameter tuning (budget: 50 configs × 1000 steps) ===\n")

results = {}

# SGD: lr × wd
print("SGD...")
best = {'loss': np.inf}
configs_sgd = []
for lr in np.logspace(-2, 1.5, 10):
    for wd in [0, 1e-5, 1e-4, 1e-3, 1e-2]:
        loss, W1, W2 = train_sgd(lr, wd)
        configs_sgd.append({'lr': lr, 'wd': wd, 'loss': loss})
        if np.isfinite(loss) and loss < best['loss']:
            best = {'lr': lr, 'wd': wd, 'loss': loss, 'W1': W1, 'W2': W2}
results['SGD'] = best
print(f"  best: lr={best['lr']:.4f} wd={best['wd']:.1e} loss={best['loss']:.4f}")

# Nesterov: lr × mu × wd
print("Nesterov...")
best = {'loss': np.inf}
configs_nest = []
for lr in np.logspace(-2, 1.5, 8):
    for mu in [0.8, 0.9, 0.95]:
        for wd in [0, 1e-4, 1e-3]:
            loss, W1, W2 = train_nesterov(lr, mu, wd)
            configs_nest.append({'lr': lr, 'mu': mu, 'wd': wd, 'loss': loss})
            if np.isfinite(loss) and loss < best['loss']:
                best = {'lr': lr, 'mu': mu, 'wd': wd, 'loss': loss, 'W1': W1, 'W2': W2}
results['Nesterov'] = best
print(f"  best: lr={best['lr']:.4f} mu={best['mu']} wd={best['wd']:.1e} loss={best['loss']:.4f}")

# AdamW: lr × (b1,b2) × wd
print("AdamW...")
best = {'loss': np.inf}
configs_adam = []
for lr in np.logspace(-3, 0.5, 10):
    for b1, b2 in [(0.9, 0.999), (0.9, 0.99), (0.95, 0.999)]:
        for wd in [0, 1e-4, 1e-3]:
            loss, W1, W2 = train_adamw(lr, b1, b2, wd)
            configs_adam.append({'lr': lr, 'b1': b1, 'b2': b2, 'wd': wd, 'loss': loss})
            if np.isfinite(loss) and loss < best['loss']:
                best = {'lr': lr, 'b1': b1, 'b2': b2, 'wd': wd, 'loss': loss, 'W1': W1, 'W2': W2}
results['AdamW'] = best
print(f"  best: lr={best['lr']:.4f} b1={best['b1']} b2={best['b2']} wd={best['wd']:.1e} loss={best['loss']:.4f}")

# Muon: lr_muon × mu_muon × wd (lr_adam from AdamW best)
print("Muon...")
best_adam_lr = results['AdamW']['lr']
best = {'loss': np.inf}
configs_muon = []
for lr_muon in np.logspace(-2.5, 0.5, 10):
    for mu_muon in [0.9, 0.95, 0.99]:
        for wd in [0, 1e-4, 1e-3]:
            loss, W1, W2 = train_muon(lr_muon, best_adam_lr, mu_muon, 0.9, 0.999, wd)
            configs_muon.append({'lr': lr_muon, 'mu': mu_muon, 'wd': wd, 'loss': loss})
            if np.isfinite(loss) and loss < best['loss']:
                best = {'lr_muon': lr_muon, 'lr_adam': best_adam_lr, 'mu': mu_muon, 'wd': wd,
                        'loss': loss, 'W1': W1, 'W2': W2}
results['Muon'] = best
print(f"  best: lr_muon={best['lr_muon']:.4f} mu={best['mu']} wd={best['wd']:.1e} loss={best['loss']:.4f}")

# ── Save best HPs ──
hp_summary = {}
for name in ['SGD', 'Nesterov', 'AdamW', 'Muon']:
    hp_summary[name] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                        for k, v in results[name].items() if k not in ('W1', 'W2')}
print(f"\n{json.dumps(hp_summary, indent=2)}")

# ── Plot: decision boundaries with best HPs ──
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
fig.subplots_adjust(wspace=0.05, top=0.82)

for i, name in enumerate(['SGD', 'Nesterov', 'AdamW', 'Muon']):
    ax = axes[i]
    W1, W2 = results[name]['W1'], results[name]['W2']
    pred = predict_grid(W1, W2)
    ax.contourf(xx, yy, pred, levels=[0,0.5,1], cmap=cmap_bg, alpha=0.85)
    ax.contour(xx, yy, pred, levels=[0.5], colors='k', linewidths=1.5, alpha=0.7)
    ax.scatter(X_all[:n_half,0], X_all[:n_half,1], c='#2060b0', s=6, alpha=0.5, edgecolors='none')
    ax.scatter(X_all[n_half:,0], X_all[n_half:,1], c='#c03040', s=6, alpha=0.5, edgecolors='none')
    ax.set_xlim(-2.8,2.8); ax.set_ylim(-2.8,2.8); ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    loss_val = results[name]['loss']
    acc_val = accuracy(W1, W2)
    ax.set_title(f'{name}\nloss={loss_val:.3f}, acc={acc_val:.1%}',
                 fontsize=11, fontweight='bold', color=COLORS[name])

fig.suptitle(f'После HP tuning ({N_TUNE} шагов, mini-batch)', fontsize=13)
fig.savefig('/root/hse26_repo/files/exp_optimizer_hptune_boundaries.pdf', bbox_inches='tight', dpi=200)
fig.savefig('/tmp/exp_optimizer_hptune_boundaries.png', bbox_inches='tight', dpi=150)
print("\nSaved boundaries")
plt.close()

# ── Plot: HP sensitivity (loss vs lr for each method) ──
fig2, axes2 = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
fig2.subplots_adjust(wspace=0.08, top=0.82)

for i, (name, configs) in enumerate([
    ('SGD', configs_sgd), ('Nesterov', configs_nest),
    ('AdamW', configs_adam), ('Muon', configs_muon),
]):
    ax = axes2[i]
    lrs = [c['lr'] if 'lr' in c else c.get('lr', 0) for c in configs]
    losses = [c['loss'] for c in configs]
    valid = [(lr, l) for lr, l in zip(lrs, losses) if np.isfinite(l) and l < 2]
    if valid:
        lrs_v, losses_v = zip(*valid)
        ax.scatter(lrs_v, losses_v, c=COLORS[name], alpha=0.5, s=20, edgecolors='none')
    best_lr = results[name].get('lr', results[name].get('lr_muon', 0))
    ax.axvline(best_lr, color=COLORS[name], ls='--', lw=1.5, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    if i == 0: ax.set_ylabel('BCE Loss')
    ax.set_title(name, fontsize=12, fontweight='bold', color=COLORS[name])
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig2.suptitle('Чувствительность к learning rate (каждая точка = 1 конфигурация)', fontsize=13)
fig2.savefig('/root/hse26_repo/files/exp_optimizer_hptune_sensitivity.pdf', bbox_inches='tight', dpi=200)
fig2.savefig('/tmp/exp_optimizer_hptune_sensitivity.png', bbox_inches='tight', dpi=150)
print("Saved sensitivity")
plt.close()
