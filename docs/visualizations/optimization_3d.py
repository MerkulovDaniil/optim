import os
import sys
import math
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, lax
import matplotlib as mpl

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

try:
    from tqdm import tqdm
except Exception:
    def tqdm(*args, **kwargs):
        class _Dummy:
            def update(self, *a, **k): pass
            def close(self): pass
        return _Dummy()



# ============================================================
# Base Config (problem, grid, view)
# ============================================================

out_dir = os.path.join(os.getcwd(), "rosenbrock")

BASE_CFG = {
    "problem": {"name": "rosenbrock", "alpha": 1.0},
    "starting_point": [0.1, 3.0],
    "grid": {
            "bounds_w1": (-2.0, 2.0),
            "bounds_w2": (-1.0, 3.0),
            "points": 200
        },
    "render": {
        "dpi": 300,
        "use_hw_accel": True,
        "zlim": [0, 2500],
        "view": {"azim": -128.0, "elev": 43.0},
        "figsize": (8, 6)
    }
}

PROB = BASE_CFG["problem"]
GRID = BASE_CFG["grid"]
VIEW = BASE_CFG["render"]["view"]
DPI = BASE_CFG["render"]["dpi"]
USE_HW = BASE_CFG["render"]["use_hw_accel"]
FIGSIZE = BASE_CFG["render"]["figsize"]
W0 = jnp.array(BASE_CFG["starting_point"])

# ============================================================
# Scenarios (your 4 requested videos)
# ============================================================

SCENARIOS = [
    {
        "title": r"GD, small stepsize: smooth and slow",
        "methods": [
            {"type": "gd", "eta": 1e-5, "name": r"GD, $\alpha=10^{-5}$", "color": "purple"},
        ],
        "fps": 30, "seconds": 20,
        "out_path": os.path.join(out_dir, "gd_small_60fps_20s.mp4"),
    },
    {
        "title": r"GD, higher stepsize (divergence)",
        # For quadratic with L=3.0, stability requires alpha < 2/L ≈ 0.666...
        # Use 0.70 to show divergence without exploding instantly.
        "methods": [
            {"type": "gd", "eta": 1e-2, "name": r"GD, $\alpha=10^{-2}$ (diverge)", "color": "crimson"},
        ],
        "fps": 2, "seconds": 10,
        "out_path": os.path.join(out_dir, "gd_diverge_2fps_10s.mp4"),
    },
    {
        "title": r"GD comparison: small vs medium",
        "methods": [
            {"type": "gd", "eta": 1e-5,  "name": r"GD, $\alpha=10^{-5}$", "color": "purple"},
            {"type": "gd", "eta": 1e-4,"name": r"GD, $\alpha=10^{-4}$",       "color": "darkgreen"},
        ],
        "fps": 2, "seconds": 20,
        "out_path": os.path.join(out_dir, "gd_small_vs_medium_2fps_20s.mp4"),
    },
    {
        "title": r"GD comparison: small vs medium vs large",
        "methods": [
            {"type": "gd", "eta": 1e-5,  "name": r"GD, $\alpha=10^{-5}$", "color": "purple"},
            {"type": "gd", "eta": 1e-4,  "name": r"GD, $\alpha=10^{-4}$", "color": "orange"},
            {"type": "gd", "eta": 1e-3, "name": r"GD, $\alpha=10^{-3}$", "color": "darkgreen"},
        ],
        "fps": 2, "seconds": 20,
        "out_path": os.path.join(out_dir, "gd_small_vs_medium_vs_large_2fps_20s.mp4"),
    },
]



# # ============================================================
# # Base Config (problem, grid, view)
# # ============================================================

# out_dir = os.path.join(os.getcwd(), "_convex_quadratic")

# BASE_CFG = {
#     "problem": {"name": "quadratic", "mu": 0, "L": 3.0},
#     "starting_point": [1.5, 3.0],
#     "grid": {
#         "bounds_w1": (-2.0, 2.0),
#         "bounds_w2": (-2.0, 4.0),
#         "points": 200
#     },
#     "render": {
#         "dpi": 300,
#         "use_hw_accel": True,
#         "zlim": [0, None],
#         "view": {"azim": -128.0, "elev": 43.0},
#         "figsize": (8, 6)
#     }
# }

# PROB = BASE_CFG["problem"]
# GRID = BASE_CFG["grid"]
# VIEW = BASE_CFG["render"]["view"]
# DPI = BASE_CFG["render"]["dpi"]
# USE_HW = BASE_CFG["render"]["use_hw_accel"]
# FIGSIZE = BASE_CFG["render"]["figsize"]
# W0 = jnp.array(BASE_CFG["starting_point"])

# # ============================================================
# # Scenarios (your 4 requested videos)
# # ============================================================


# mu, L = PROB["mu"], PROB["L"]

# SCENARIOS = [
#     {
#         "title": r"GD, small stepsize: smooth and slow",
#         "methods": [
#             {"type": "gd", "eta": 3e-3, "name": r"GD, $\alpha=3\cdot 10^{-3}$", "color": "purple"},
#         ],
#         "fps": 60, "seconds": 20,
#         "out_path": os.path.join(out_dir, "gd_small_60fps_20s.mp4"),
#     },
#     {
#         "title": r"GD, optimal stepsize $\alpha^*=\frac{1}{L}$",
#         "methods": [
#             {"type": "gd", "eta": "auto", "name": r"GD, $\alpha^*$", "color": "blue"},
#         ],
#         "fps": 4, "seconds": 100,
#         "out_path": os.path.join(out_dir, "gd_opt_4fps_100s.mp4"),
#     },
#     {
#         "title": r"GD, higher stepsize (divergence)",
#         # For quadratic with L=3.0, stability requires alpha < 2/L ≈ 0.666...
#         # Use 0.70 to show divergence without exploding instantly.
#         "methods": [
#             {"type": "gd", "eta": 0.70, "name": r"GD, $\alpha=0.70$ (diverge)", "color": "crimson"},
#         ],
#         "fps": 2, "seconds": 10,
#         "out_path": os.path.join(out_dir, "gd_diverge_2fps_10s.mp4"),
#     },
#     {
#         "title": r"GD comparison: small vs optimal",
#         "methods": [
#             {"type": "gd", "eta": 3e-3,  "name": r"GD, $\alpha=3\cdot 10^{-3}$", "color": "purple"},
#             {"type": "gd", "eta": "auto","name": r"GD, $\alpha^*$",       "color": "darkgreen"},
#         ],
#         "fps": 2, "seconds": 20,
#         "out_path": os.path.join(out_dir, "gd_small_vs_opt_2fps_20s.mp4"),
#     },
#     {
#         "title": r"GD comparison: small vs medium vs optimal",
#         "methods": [
#             {"type": "gd", "eta": 3e-3,  "name": r"GD, $\alpha=3\cdot 10^{-3}$", "color": "purple"},
#             {"type": "gd", "eta": 1e-1,  "name": r"GD, $\alpha=10^{-1}$", "color": "orange"},
#             {"type": "gd", "eta": "auto","name": r"GD, $\alpha^*$",       "color": "darkgreen"},
#         ],
#         "fps": 2, "seconds": 20,
#         "out_path": os.path.join(out_dir, "gd_small_vs_medium_vs_opt_2fps_20s.mp4"),
#     },
# ]

# # ============================================================
# # Base Config (problem, grid, view)
# # ============================================================

# out_dir = os.path.join(os.getcwd(), "strongly_convex_quadratic")

# BASE_CFG = {
#     "problem": {"name": "quadratic", "mu": 2, "L": 3.0},
#     "starting_point": [1.5, 3.0],
#     "grid": {
#         "bounds_w1": (-2.0, 4.0),
#         "bounds_w2": (-2.0, 4.0),
#         "points": 200
#     },
#     "render": {
#         "dpi": 300,
#         "use_hw_accel": True,
#         "zlim": [0, None],
#         "view": {"azim": -128.0, "elev": 43.0},
#         "figsize": (8, 6)
#     }
# }

# PROB = BASE_CFG["problem"]
# GRID = BASE_CFG["grid"]
# VIEW = BASE_CFG["render"]["view"]
# DPI = BASE_CFG["render"]["dpi"]
# USE_HW = BASE_CFG["render"]["use_hw_accel"]
# FIGSIZE = BASE_CFG["render"]["figsize"]
# W0 = jnp.array(BASE_CFG["starting_point"])

# mu, L = PROB["mu"], PROB["L"]


# SCENARIOS = [
#     {
#         "title": r"GD, small stepsize: smooth and slow",
#         "methods": [
#             {"type": "gd", "eta": 3e-3, "name": r"GD, $\alpha=3\cdot 10^{-3}$", "color": "purple"},
#         ],
#         "fps": 60, "seconds": 20,
#         "out_path": os.path.join(out_dir, "gd_small_60fps_20s.mp4"),
#     },
#     {
#         "title": r"GD, optimal stepsize $\alpha^*=\frac{2}{\mu+L}$",
#         "methods": [
#             {"type": "gd", "eta": "auto", "name": r"GD, $\alpha^*$", "color": "blue"},
#         ],
#         "fps": 2, "seconds": 20,
#         "out_path": os.path.join(out_dir, "gd_opt_2fps_20s.mp4"),
#     },
#     {
#         "title": r"GD, higher stepsize (divergence)",
#         # For quadratic with L=3.0, stability requires alpha < 2/L ≈ 0.666...
#         # Use 0.70 to show divergence without exploding instantly.
#         "methods": [
#             {"type": "gd", "eta": 0.70, "name": r"GD, $\alpha=0.70$ (diverge)", "color": "crimson"},
#         ],
#         "fps": 2, "seconds": 10,
#         "out_path": os.path.join(out_dir, "gd_diverge_2fps_10s.mp4"),
#     },
#     {
#         "title": r"GD comparison: small vs optimal",
#         "methods": [
#             {"type": "gd", "eta": 3e-3,  "name": r"GD, $\alpha=3\cdot 10^{-3}$", "color": "purple"},
#             {"type": "gd", "eta": "auto","name": r"GD, $\alpha^*$",       "color": "darkgreen"},
#         ],
#         "fps": 2, "seconds": 20,
#         "out_path": os.path.join(out_dir, "gd_small_vs_opt_2fps_20s.mp4"),
#     },
#     {
#         "title": r"GD comparison: small vs medium vs optimal",
#         "methods": [
#             {"type": "gd", "eta": 3e-3,  "name": r"GD, $\alpha=3\cdot 10^{-3}$", "color": "purple"},
#             {"type": "gd", "eta": 1e-1,  "name": r"GD, $\alpha=10^{-1}$", "color": "orange"},
#             {"type": "gd", "eta": "auto","name": r"GD, $\alpha^*$",       "color": "darkgreen"},
#         ],
#         "fps": 2, "seconds": 20,
#         "out_path": os.path.join(out_dir, "gd_small_vs_medium_vs_opt_2fps_20s.mp4"),
#     },
# ]

# # ============================================================
# # Base Config (problem, grid, view)
# # ============================================================

# out_dir = os.path.join(os.getcwd(), "strongly_convex_isotropic")

# BASE_CFG = {
#     "problem": {"name": "quadratic", "mu": 3, "L": 3.0},
#     "starting_point": [1.5, 3.0],
#     "grid": {
#         "bounds_w1": (-2.0, 4.0),
#         "bounds_w2": (-2.0, 4.0),
#         "points": 200
#     },
#     "render": {
#         "dpi": 300,
#         "use_hw_accel": True,
#         "zlim": [0, None],
#         "view": {"azim": -128.0, "elev": 43.0},
#         "figsize": (8, 6)
#     }
# }

# PROB = BASE_CFG["problem"]
# GRID = BASE_CFG["grid"]
# VIEW = BASE_CFG["render"]["view"]
# DPI = BASE_CFG["render"]["dpi"]
# USE_HW = BASE_CFG["render"]["use_hw_accel"]
# FIGSIZE = BASE_CFG["render"]["figsize"]
# W0 = jnp.array(BASE_CFG["starting_point"])

# mu, L = PROB["mu"], PROB["L"]


# SCENARIOS = [
#     {
#         "title": r"GD, small stepsize: smooth and slow",
#         "methods": [
#             {"type": "gd", "eta": 3e-3, "name": r"GD, $\alpha=3\cdot 10^{-3}$", "color": "purple"},
#         ],
#         "fps": 60, "seconds": 20,
#         "out_path": os.path.join(out_dir, "gd_small_60fps_20s.mp4"),
#     },
#     {
#         "title": r"GD, optimal stepsize $\alpha^*=\frac{2}{\mu+L}$",
#         "methods": [
#             {"type": "gd", "eta": "auto", "name": r"GD, $\alpha^*$", "color": "blue"},
#         ],
#         "fps": 2, "seconds": 20,
#         "out_path": os.path.join(out_dir, "gd_opt_2fps_20s.mp4"),
#     },
#     {
#         "title": r"GD, higher stepsize (divergence)",
#         # For quadratic with L=3.0, stability requires alpha < 2/L ≈ 0.666...
#         # Use 0.70 to show divergence without exploding instantly.
#         "methods": [
#             {"type": "gd", "eta": 0.70, "name": r"GD, $\alpha=0.70$ (diverge)", "color": "crimson"},
#         ],
#         "fps": 2, "seconds": 10,
#         "out_path": os.path.join(out_dir, "gd_diverge_2fps_10s.mp4"),
#     },
#     {
#         "title": r"GD comparison: small vs optimal",
#         "methods": [
#             {"type": "gd", "eta": 3e-3,  "name": r"GD, $\alpha=3\cdot 10^{-3}$", "color": "purple"},
#             {"type": "gd", "eta": "auto","name": r"GD, $\alpha^*$",       "color": "darkgreen"},
#         ],
#         "fps": 2, "seconds": 20,
#         "out_path": os.path.join(out_dir, "gd_small_vs_opt_2fps_20s.mp4"),
#     },
#     {
#         "title": r"GD comparison: small vs medium vs optimal",
#         "methods": [
#             {"type": "gd", "eta": 3e-3,  "name": r"GD, $\alpha=3\cdot 10^{-3}$", "color": "purple"},
#             {"type": "gd", "eta": 1e-1,  "name": r"GD, $\alpha=10^{-1}$", "color": "orange"},
#             {"type": "gd", "eta": "auto","name": r"GD, $\alpha^*$",       "color": "darkgreen"},
#         ],
#         "fps": 2, "seconds": 20,
#         "out_path": os.path.join(out_dir, "gd_small_vs_medium_vs_opt_2fps_20s.mp4"),
#     },
# ]

# # ============================================================
# # Base Config (problem, grid, view)
# # ============================================================

# out_dir = os.path.join(os.getcwd(), "convex_agd")

# BASE_CFG = {
#     "problem": {"name": "quadratic", "mu": 1, "L": 3.0},
#     "starting_point": [1.5, 3.0],
#     "grid": {
#         "bounds_w1": (-2.0, 4.0),
#         "bounds_w2": (-2.0, 4.0),
#         "points": 200
#     },
#     "render": {
#         "dpi": 300,
#         "use_hw_accel": True,
#         "zlim": [0, None],
#         "view": {"azim": -128.0, "elev": 43.0},
#         "figsize": (8, 6)
#     }
# }

# PROB = BASE_CFG["problem"]
# GRID = BASE_CFG["grid"]
# VIEW = BASE_CFG["render"]["view"]
# DPI = BASE_CFG["render"]["dpi"]
# USE_HW = BASE_CFG["render"]["use_hw_accel"]
# FIGSIZE = BASE_CFG["render"]["figsize"]
# W0 = jnp.array(BASE_CFG["starting_point"])

# mu, L = PROB["mu"], PROB["L"]


# SCENARIOS = [
#     {
#         "title": r"Small stepsize: smooth and slow (GD / HB / NAG)",
#         "methods": [
#             {"type": "gd",  "eta": 3e-3,   "name": r"GD, $\alpha=3\cdot 10^{-3}$", "color": "purple"},
#             {"type": "hb",  "eta": "auto", "beta": "auto", "scale_eta": 0.1,
#              "name": r"HB, $\eta=\eta^* \times 0.1,\ \beta=\beta^*$", "color": "orange"},
#             {"type": "nag", "eta": "auto", "beta": "auto", "scale_eta": 0.1,
#              "name": r"NAG, $\eta=\eta^*\!\times 0.1,\ \beta=\beta^*$", "color": "teal"},
#         ],
#         "fps": 60, "seconds": 20,
#         "out_path": os.path.join(out_dir, "gd_hb_nag_small_60fps_20s.mp4"),
#     },
#     {
#         "title": r"Optimal parameters (GD / HB / NAG): $\alpha^*=\frac{2}{\mu+L}$, $(\eta^*,\beta^*)$",
#         "methods": [
#             {"type": "gd",  "eta": "auto", "name": r"GD, $\alpha^*$", "color": "blue"},
#             {"type": "hb",  "eta": "auto", "beta": "auto",
#              "name": r"HB, $(\eta^*,\beta^*)$", "color": "crimson"},
#             {"type": "nag", "eta": "auto", "beta": "auto",
#              "name": r"NAG, $(\eta^*,\beta^*)$", "color": "darkgreen"},
#         ],
#         "fps": 2, "seconds": 20,
#         "out_path": os.path.join(out_dir, "gd_hb_nag_opt_2fps_20s.mp4"),
#     },
# ]


# ============================================================
# Objectives
# ============================================================

def rosenbrock(w):
    w1, w2 = w
    return (1.0 - w1) ** 2 + 100.0 * (w2 - w1 ** 2) ** 2

def quadratic_factory(mu: float, L: float):
    Q = jnp.array([[L, 0.0], [0.0, mu]])
    b = jnp.zeros(2)
    c = 0.0
    def f(w):
        return 0.5 * w @ Q @ w - b @ w + c
    return f

prob_name = PROB["name"].lower()
if prob_name == "quadratic":
    f = quadratic_factory(PROB["mu"], PROB["L"])
elif prob_name == "rosenbrock":
    f = rosenbrock
else:
    raise ValueError("Unsupported problem: choose 'rosenbrock' or 'quadratic'")

# JITed grad
g = jit(grad(f))

# ============================================================
# Optimizers (GD + optional HB/NAG helpers)
# ============================================================

def gd_eta_opt(mu, L):
    if mu == 0:
        return 1.0 / L
    else:
        return 2.0 / (mu + L)

@jit
def gd_update(w, m, eta):
    return w - eta * g(w), m

@jit
def hb_update(w, m, eta, beta):
    m_next = beta * m + g(w)
    return w - eta * m_next, m_next

@jit
def nag_update(w, w_prev, eta, beta):
    y = w + beta * (w - w_prev)
    w_next = y - eta * g(y)
    return w_next, w

@dataclass
class OptimSpec:
    name: str
    color: str
    kind: str  # "gd" | "hb" | "nag"
    params: tuple = ()

def hb_params_opt(mu, L):
    sL, sm = jnp.sqrt(L), jnp.sqrt(mu)
    eta = 4.0 / (sL + sm) ** 2
    beta = ((sL - sm) / (sL + sm)) ** 2
    return float(eta), float(beta)

def nag_params_opt(mu, L):
    kappa = L / mu
    q = (jnp.sqrt(kappa) - 1.0) / (jnp.sqrt(kappa) + 1.0)
    beta = float(q)
    eta = 1.0 / L
    return float(eta), float(beta)

def build_optimizers(methods: list) -> list:
    out = []
    mu, L = PROB.get("mu", None), PROB.get("L", None)
    for m in methods:
        t = m["type"].lower()
        name = m.get("name", t.upper())
        color = m.get("color", "black")
        scale = float(m.get("scale_eta", 1.0))

        if t == "gd":
            eta = m.get("eta", "auto")
            if eta == "auto":
                if prob_name != "quadratic":
                    raise ValueError("Auto GD params only defined for quadratic.")
                eta = gd_eta_opt(mu, L)
            eta = float(eta) * scale
            out.append(OptimSpec(name, color, "gd", (eta,)))

        elif t == "hb":
            eta = m.get("eta", "auto")
            beta = m.get("beta", "auto")
            if eta == "auto" or beta == "auto":
                if prob_name != "quadratic":
                    raise ValueError("Auto HB params only defined for quadratic.")
                eta_star, beta_star = hb_params_opt(mu, L)
                if eta == "auto":  eta = eta_star
                if beta == "auto": beta = beta_star
            eta = float(eta) * scale
            beta = float(beta)
            out.append(OptimSpec(name, color, "hb", (eta, beta)))

        elif t == "nag":
            eta = m.get("eta", "auto")
            beta = m.get("beta", "auto")
            if eta == "auto" or beta == "auto":
                if prob_name != "quadratic":
                    raise ValueError("Auto NAG params only defined for quadratic.")
                eta_star, beta_star = nag_params_opt(mu, L)
                if eta == "auto":  eta = eta_star
                if beta == "auto": beta = beta_star
            eta = float(eta) * scale
            beta = float(beta)
            out.append(OptimSpec(name, color, "nag", (eta, beta)))

        else:
            raise ValueError(f"Unknown method type: {t}")

    return out


def run_one(opt: OptimSpec, w0: jnp.ndarray, n_steps: int):
    if opt.kind == "nag":
        eta, beta = opt.params
        def step(carry, _):
            w, w_prev = carry
            w_next, w_prev_next = nag_update(w, w_prev, eta, beta)
            return (w_next, w_prev_next), w_next
        init = (w0, w0)
        (_, _), ws = lax.scan(step, init, xs=None, length=n_steps)
        ws_full = jnp.concatenate([w0[None, :], ws], axis=0)
        return ws_full
    else:
        def step(carry, _):
            w, m = carry
            if opt.kind == "gd":
                (eta,) = opt.params
                w_next, m_next = gd_update(w, m, eta)
            elif opt.kind == "hb":
                eta, beta = opt.params
                w_next, m_next = hb_update(w, m, eta, beta)
            else:
                w_next, m_next = w, m
            return (w_next, m_next), w_next
        init = (w0, jnp.zeros_like(w0))
        (_, _), ws = lax.scan(step, init, xs=None, length=n_steps)
        ws_full = jnp.concatenate([w0[None, :], ws], axis=0)
        return ws_full

# ============================================================
# Surface precompute (once)
# ============================================================

x = jnp.linspace(GRID["bounds_w1"][0], GRID["bounds_w1"][1], GRID["points"])
y = jnp.linspace(GRID["bounds_w2"][0], GRID["bounds_w2"][1], GRID["points"])
X, Y = jnp.meshgrid(x, y)
points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
Z = vmap(f)(points).reshape(X.shape)
Z_min = float(jnp.min(Z))
Z_max = float(jnp.max(Z))

# ============================================================
# Rendering helper
# ============================================================

def pick_writer(fps: int):
    codec = 'libx264'
    extra = ['-pix_fmt', 'yuv420p', '-crf', '22']  # quality/size
    if USE_HW and sys.platform == 'darwin':
        codec = 'h264_videotoolbox'
        extra = ['-b:v', '5M']
    return FFMpegWriter(fps=fps, codec=codec, extra_args=extra)

def run_scenario(title_suffix: str, methods: list, fps: int, seconds: int, out_path: str):
    """
    methods: list of dicts like {"type":"gd","eta":1e-3,"name":"...","color":"..."}
    fps, seconds define exact frames; iterations = frames - 1
    """
    n_frames = int(fps * seconds)
    n_steps = max(0, n_frames - 1)

    optimizers = build_optimizers(methods)

    # simulate
    trajs = []
    for o in optimizers:
        ws = run_one(o, W0, n_steps)
        trajs.append(ws)
    trajectories = jnp.stack(trajs)  # [n_opt, n_steps+1, 2]
    traj_vals = vmap(vmap(f))(trajectories)  # [n_opt, n_steps+1]

    # Matplotlib 3D figure
    mpl.rcParams['figure.dpi'] = DPI
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['grid.linestyle'] = ':'

    fig, ax = plt.subplots(
        subplot_kw=dict(projection='3d', azim=VIEW["azim"], elev=VIEW["elev"], computed_zorder=False),
        figsize=FIGSIZE
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.92)

    norm = mpl.colors.LogNorm(vmin=max(Z_min, 1e-8), vmax=Z_max)
    surf = ax.plot_surface(X, Y, Z, alpha=0.6, cmap='coolwarm', norm=norm)

    ax.set_xlabel(r'$w_1$'); ax.set_ylabel(r'$w_2$')
    ax.set_xlim(GRID["bounds_w1"]); ax.set_ylim(GRID["bounds_w2"])

    zlim_cfg = BASE_CFG["render"].get("zlim")
    if zlim_cfg is not None:
        lower, upper = zlim_cfg
        if lower is None: lower = Z_min
        if upper is None: upper = Z_max
        if upper <= lower: upper = lower + max(1.0, Z_max - Z_min)
        ax.set_zlim(lower, upper)
    else:
        ax.set_zlim(0, Z_max)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$f(w)$', rotation=0)

    if prob_name == "quadratic":
        mu, L = PROB["mu"], PROB["L"]
        base_title = f"Quadratic: $\\mu={mu}$, $L={L}$"
        opt_w = jnp.zeros(2)
    else:
        base_title = prob_name.capitalize()
        opt_w = jnp.array([1.0, 1.0])
    ax.scatter(opt_w[0], opt_w[1], f(opt_w), color='gold', marker='*', s=300, label='Optimal solution')
    ax.set_title(f"{base_title} — {title_suffix}")

    # Artists
    lines, markers = [], []
    for o in optimizers:
        (line,) = ax.plot([], [], [], color=o.color, label=o.name)
        lines.append(line)
        sc = ax.scatter([], [], [], color=o.color, s=40)
        markers.append(sc)

    leg = ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), borderaxespad=0.1,
                    frameon=True, framealpha=0.9)

    # Convert to host arrays once
    trajectories_np = jnp.asarray(trajectories).copy().block_until_ready()
    traj_vals_np = jnp.asarray(traj_vals).copy().block_until_ready()

    def init_anim():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        for sc in markers:
            sc._offsets3d = (jnp.array([]), jnp.array([]), jnp.array([]))
        return [*lines, *markers]

    def animate(i):
        idx = i  # 0..n_frames-1
        for k, (line, sc) in enumerate(zip(lines, markers)):
            ws = trajectories_np[k, : idx + 1]
            fsel = traj_vals_np[k, : idx + 1]
            line.set_data(ws[:, 0], ws[:, 1])
            line.set_3d_properties(fsel)
            cur = trajectories_np[k, idx]
            sc._offsets3d = (
                jnp.array([cur[0]]),
                jnp.array([cur[1]]),
                jnp.array([traj_vals_np[k, idx]])
            )
        return [*lines, *markers]

    interval_ms = int(round(1000.0 / fps))
    ani = FuncAnimation(fig, animate, init_func=init_anim,
                        interval=interval_ms, frames=n_frames, blit=False)

    # Save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = pick_writer(fps)
    bar = tqdm(total=n_frames, desc=f"Saving: {os.path.basename(out_path)}", unit="frame")

    def _progress_callback(i, n):
        bar.update(1)
        if i + 1 >= n:
            bar.close()

    ani.save(out_path, writer=writer, dpi=DPI,
             progress_callback=_progress_callback, savefig_kwargs={"pad_inches": 0.02})
    plt.close(fig)




from concurrent.futures import ProcessPoolExecutor, as_completed

def render_scenario(sc):
    run_scenario(
        title_suffix=sc["title"],
        methods=sc["methods"],
        fps=sc["fps"],
        seconds=sc["seconds"],
        out_path=sc["out_path"],
    )
    return sc["out_path"]

if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    # Run all scenarios in parallel processes
    max_workers = min(len(SCENARIOS), max(1, (os.cpu_count() or 2) // 2))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(render_scenario, sc) for sc in SCENARIOS]
        for fut in as_completed(futs):
            try:
                print("Done:", fut.result())
            except Exception as e:
                print("Failed:", e)