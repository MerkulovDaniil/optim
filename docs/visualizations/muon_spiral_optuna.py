"""
Optuna HP tuning for SGD, Nesterov, AdamW, Muon on spiral classification.
Each method gets 100 trials × 1000 steps. All relevant HPs are tuned.
"""
import numpy as np
import optuna
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import json, warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})
COLORS = {'SGD': '#4C72B0', 'Nesterov': '#DD8452', 'AdamW': '#55A868', 'Muon': '#8172B3'}
cmap_bg = ListedColormap(['#dce6f5', '#f5dce0'])

# ── Data ──
np.random.seed(42)
n_pts = 400; n_half = n_pts // 2; batch_size = 40
theta = np.linspace(0.5, 3.2*np.pi, n_half)
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
    Z = np.tanh(W1 @ X); logits = (W2 @ Z).ravel()
    return Z, logits, sigmoid(logits)

def bce_loss(y_hat, y):
    y_hat = np.clip(y_hat, 1e-7, 1-1e-7)
    return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

def grads(W1, W2, X, y):
    n = X.shape[1]; Z, _, y_hat = forward(W1, W2, X)
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
        A_ = X @ X.T; X = a*X + b*A_@X + c*(A_@A_)@X
    return X.T if transposed else X

np.random.seed(7)
W1_init = np.random.randn(h, 3) * 0.15
W2_init = np.random.randn(1, h) * 0.15

def get_batch(rng):
    return X_aug[:, rng.choice(n_pts, batch_size, replace=False)], y_all[rng.choice(n_pts, batch_size, replace=False)]

N_STEPS = 1000

# ── Training functions ──
def train_sgd(lr, momentum, wd, seed=0):
    W1, W2 = W1_init.copy(), W2_init.copy()
    V1, V2 = np.zeros_like(W1), np.zeros_like(W2)
    rng = np.random.RandomState(seed)
    for _ in range(N_STEPS):
        idx = rng.choice(n_pts, batch_size, replace=False)
        dW1, dW2 = grads(W1, W2, X_aug[:, idx], y_all[idx])
        V1 = momentum*V1 + dW1 + wd*W1; V2 = momentum*V2 + dW2 + wd*W2
        W1 -= lr*V1; W2 -= lr*V2
    return full_loss(W1, W2), W1, W2

def train_nesterov(lr, momentum, wd, seed=0):
    W1, W2 = W1_init.copy(), W2_init.copy()
    V1, V2 = np.zeros_like(W1), np.zeros_like(W2)
    rng = np.random.RandomState(seed)
    for _ in range(N_STEPS):
        idx = rng.choice(n_pts, batch_size, replace=False)
        dW1, dW2 = grads(W1+momentum*V1, W2+momentum*V2, X_aug[:, idx], y_all[idx])
        V1 = momentum*V1 - lr*(dW1+wd*W1); V2 = momentum*V2 - lr*(dW2+wd*W2)
        W1 += V1; W2 += V2
    return full_loss(W1, W2), W1, W2

def train_adamw(lr, b1, b2, wd, eps, seed=0):
    W1, W2 = W1_init.copy(), W2_init.copy()
    m1,v1 = np.zeros_like(W1), np.zeros_like(W1)
    m2,v2 = np.zeros_like(W2), np.zeros_like(W2)
    rng = np.random.RandomState(seed)
    for i in range(N_STEPS):
        t = i+1; idx = rng.choice(n_pts, batch_size, replace=False)
        dW1, dW2 = grads(W1, W2, X_aug[:, idx], y_all[idx])
        for W, dW, m, v in [(W1,dW1,m1,v1),(W2,dW2,m2,v2)]:
            m[:] = b1*m + (1-b1)*dW; v[:] = b2*v + (1-b2)*dW**2
            mh = m/(1-b1**t); vh = v/(1-b2**t)
            W[:] = W*(1-lr*wd) - lr*mh/(np.sqrt(vh)+eps)
    return full_loss(W1, W2), W1, W2

def train_muon(lr_muon, mu_muon, ns_steps, lr_adam, b1, b2, wd, eps, seed=0):
    W1, W2 = W1_init.copy(), W2_init.copy()
    Buf1 = np.zeros_like(W1)
    m2,v2 = np.zeros_like(W2), np.zeros_like(W2)
    rng = np.random.RandomState(seed)
    for i in range(N_STEPS):
        t = i+1; idx = rng.choice(n_pts, batch_size, replace=False)
        dW1, dW2 = grads(W1, W2, X_aug[:, idx], y_all[idx])
        Buf1 = mu_muon*Buf1 + dW1
        O = newton_schulz(mu_muon*Buf1 + dW1, steps=ns_steps)
        W1 = W1*(1-lr_muon*wd) - lr_muon*O
        m2[:] = b1*m2 + (1-b1)*dW2; v2[:] = b2*v2 + (1-b2)*dW2**2
        mh = m2/(1-b1**t); vh = v2/(1-b2**t)
        W2[:] = W2*(1-lr_adam*wd) - lr_adam*mh/(np.sqrt(vh)+eps)
    return full_loss(W1, W2), W1, W2

# ── Optuna objectives ──
N_TRIALS = 500

def obj_sgd(trial):
    lr = trial.suggest_float('lr', 1e-3, 30, log=True)
    momentum = trial.suggest_float('momentum', 0.0, 0.99)
    wd = trial.suggest_float('wd', 1e-6, 0.1, log=True)
    loss, _, _ = train_sgd(lr, momentum, wd)
    return loss if np.isfinite(loss) else 10.0

def obj_nesterov(trial):
    lr = trial.suggest_float('lr', 1e-3, 30, log=True)
    momentum = trial.suggest_float('momentum', 0.5, 0.99)
    wd = trial.suggest_float('wd', 1e-6, 0.1, log=True)
    loss, _, _ = train_nesterov(lr, momentum, wd)
    return loss if np.isfinite(loss) else 10.0

def obj_adamw(trial):
    lr = trial.suggest_float('lr', 1e-4, 3.0, log=True)
    b1 = trial.suggest_float('beta1', 0.8, 0.99)
    b2 = trial.suggest_float('beta2', 0.9, 0.9999, log=True)
    wd = trial.suggest_float('wd', 1e-6, 0.1, log=True)
    eps = trial.suggest_float('eps', 1e-10, 1e-4, log=True)
    loss, _, _ = train_adamw(lr, b1, b2, wd, eps)
    return loss if np.isfinite(loss) else 10.0

def obj_muon(trial):
    lr_muon = trial.suggest_float('lr_muon', 1e-3, 3.0, log=True)
    mu_muon = trial.suggest_float('mu_muon', 0.8, 0.999)
    ns_steps = trial.suggest_int('ns_steps', 3, 10)
    lr_adam = trial.suggest_float('lr_adam', 1e-4, 3.0, log=True)
    b1 = trial.suggest_float('beta1', 0.8, 0.99)
    b2 = trial.suggest_float('beta2', 0.9, 0.9999, log=True)
    wd = trial.suggest_float('wd', 1e-6, 0.1, log=True)
    eps = trial.suggest_float('eps', 1e-10, 1e-4, log=True)
    loss, _, _ = train_muon(lr_muon, mu_muon, ns_steps, lr_adam, b1, b2, wd, eps)
    return loss if np.isfinite(loss) else 10.0

# ── Run Optuna ──
print(f"=== Optuna HP tuning: {N_TRIALS} trials × {N_STEPS} steps ===\n")

studies = {}
for name, obj_fn in [('SGD', obj_sgd), ('Nesterov', obj_nesterov),
                      ('AdamW', obj_adamw), ('Muon', obj_muon)]:
    print(f"{name}...", end=' ', flush=True)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj_fn, n_trials=N_TRIALS, show_progress_bar=False)
    studies[name] = study
    print(f"best={study.best_value:.4f}  params={study.best_params}")

# ── Retrain best configs & get weights ──
results = {}
for name in ['SGD', 'Nesterov', 'AdamW', 'Muon']:
    p = studies[name].best_params
    if name == 'SGD':
        loss, W1, W2 = train_sgd(p['lr'], p['momentum'], p['wd'])
    elif name == 'Nesterov':
        loss, W1, W2 = train_nesterov(p['lr'], p['momentum'], p['wd'])
    elif name == 'AdamW':
        loss, W1, W2 = train_adamw(p['lr'], p['beta1'], p['beta2'], p['wd'], p['eps'])
    else:
        loss, W1, W2 = train_muon(p['lr_muon'], p['mu_muon'], p['ns_steps'],
                                   p['lr_adam'], p['beta1'], p['beta2'], p['wd'], p['eps'])
    results[name] = {'loss': loss, 'acc': accuracy(W1, W2), 'W1': W1, 'W2': W2, 'params': p}

print("\n=== Best configurations ===")
for name in ['SGD', 'Nesterov', 'AdamW', 'Muon']:
    r = results[name]
    print(f"\n{name}: loss={r['loss']:.4f}, acc={r['acc']:.1%}")
    for k, v in r['params'].items():
        print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")

# ── Plot 1: Decision boundaries ──
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
    ax.set_title(f'{name}\nloss={results[name]["loss"]:.3f}, acc={results[name]["acc"]:.1%}',
                 fontsize=11, fontweight='bold', color=COLORS[name])

fig.suptitle(f'Optuna HP tuning ({N_TRIALS} trials, {N_STEPS} steps)', fontsize=13)
fig.savefig('/root/hse26_repo/files/exp_optuna_boundaries.pdf', bbox_inches='tight', dpi=200)
fig.savefig('/tmp/exp_optuna_boundaries.png', bbox_inches='tight', dpi=150)
print("\nSaved boundaries")
plt.close()

# ── Plot 2: Optimization history ──
fig2, axes2 = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
fig2.subplots_adjust(wspace=0.08, top=0.82)

for i, name in enumerate(['SGD', 'Nesterov', 'AdamW', 'Muon']):
    ax = axes2[i]
    study = studies[name]
    vals = [t.value for t in study.trials if t.value is not None and t.value < 2]
    best_so_far = np.minimum.accumulate(vals) if vals else []
    ax.scatter(range(len(vals)), vals, c=COLORS[name], alpha=0.3, s=12, edgecolors='none')
    if len(best_so_far):
        ax.plot(best_so_far, color=COLORS[name], lw=2.5)
    ax.set_xlabel('Trial')
    if i == 0: ax.set_ylabel('BCE Loss')
    ax.set_title(name, fontsize=12, fontweight='bold', color=COLORS[name])
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig2.suptitle(f'Optuna optimization history ({N_TRIALS} trials)', fontsize=13)
fig2.savefig('/root/hse26_repo/files/exp_optuna_history.pdf', bbox_inches='tight', dpi=200)
fig2.savefig('/tmp/exp_optuna_history.png', bbox_inches='tight', dpi=150)
print("Saved history")
plt.close()

# ── Plot 3: Param importances ──
fig3, axes3 = plt.subplots(1, 4, figsize=(14, 3.5))
fig3.subplots_adjust(wspace=0.35, top=0.82)

for i, name in enumerate(['SGD', 'Nesterov', 'AdamW', 'Muon']):
    ax = axes3[i]
    try:
        imp = optuna.importance.get_param_importances(studies[name])
        params = list(imp.keys())
        values = list(imp.values())
        bars = ax.barh(range(len(params)), values, color=COLORS[name], alpha=0.8)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params, fontsize=9)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
    except Exception as e:
        ax.text(0.5, 0.5, str(e)[:50], transform=ax.transAxes, ha='center', fontsize=8)
    ax.set_title(name, fontsize=12, fontweight='bold', color=COLORS[name])
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig3.suptitle('Hyperparameter importance (fANOVA)', fontsize=13)
fig3.savefig('/root/hse26_repo/files/exp_optuna_importance.pdf', bbox_inches='tight', dpi=200)
fig3.savefig('/tmp/exp_optuna_importance.png', bbox_inches='tight', dpi=150)
print("Saved importance")
plt.close()
