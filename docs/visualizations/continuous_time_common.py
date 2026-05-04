from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.integrate import solve_ivp


HERE = Path(__file__).resolve().parent
LECTURES = HERE
DATA = HERE
OPTIM_VIS = HERE

COLORS = {
    "gf": "#19A0FF",
    "agf": "#FF3EA5",
    "gd": "#FFD23F",
    "hb": "#FF5A5F",
    "nag": "#00D084",
    "ink": "#0B1020",
    "muted": "#697386",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save_pdf(fig: plt.Figure, path: Path, *, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight" if tight else None, pad_inches=0.02)
    plt.close(fig)
    print(f"saved {path}")


def quadratic_problem():
    theta = np.deg2rad(32)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    hessian = rot @ np.diag([1.0, 80.0]) @ rot.T
    x0 = np.array([3.2, 2.1])

    def grad(x):
        return hessian @ x

    def gf_rhs(_t, x):
        return -grad(x)

    def agf_rhs(t, state):
        x, v = state[:2], state[2:]
        return np.r_[v, -3.0 / max(t, 0.05) * v - grad(x)]

    t = np.linspace(0.05, 9.0, 900)
    gf = solve_ivp(gf_rhs, (t[0], t[-1]), x0, t_eval=t, rtol=1e-9, atol=1e-11).y.T
    agf = solve_ivp(agf_rhs, (t[0], t[-1]), np.r_[x0, [0.0, 0.0]], t_eval=t, max_step=0.02, rtol=1e-9, atol=1e-11).y[:2].T

    alpha = 1.86 / 80.0
    x = x0.copy()
    gd = [x.copy()]
    for _ in range(420):
        x = x - alpha * grad(x)
        gd.append(x.copy())
    gd = np.asarray(gd)

    xg = np.linspace(-3.8, 3.8, 260)
    yg = np.linspace(-2.8, 2.8, 260)
    xx, yy = np.meshgrid(xg, yg)
    zz = 0.5 * (hessian[0, 0] * xx**2 + 2 * hessian[0, 1] * xx * yy + hessian[1, 1] * yy**2)
    return {
        "title": r"ill-conditioned quadratic, $\kappa=80$",
        "xlim": (-3.8, 3.8),
        "ylim": (-2.8, 2.8),
        "grid": (xx, yy, zz),
        "trajs": {"GF": gf, "AGF": agf, "GD": gd},
        "colors": {"GF": COLORS["gf"], "AGF": COLORS["agf"], "GD": COLORS["gd"]},
        "minimizer": (0.0, 0.0),
    }


def rosenbrock_problem():
    scale = 6.0
    x0 = np.array([-1.25, 1.35])

    def grad(x):
        return np.array([-2 * (1 - x[0]) - 4 * scale * x[0] * (x[1] - x[0] ** 2), 2 * scale * (x[1] - x[0] ** 2)])

    def gf_rhs(_t, x):
        return -grad(x)

    def agf_rhs(t, state):
        x, v = state[:2], state[2:]
        return np.r_[v, -3.0 / max(t, 0.08) * v - grad(x)]

    t = np.linspace(0.08, 6.0, 1100)
    gf = solve_ivp(gf_rhs, (t[0], t[-1]), x0, t_eval=t, max_step=0.01, rtol=1e-8, atol=1e-10).y.T
    agf = solve_ivp(agf_rhs, (t[0], t[-1]), np.r_[x0, [0.0, 0.0]], t_eval=t, max_step=0.006, rtol=1e-8, atol=1e-10).y[:2].T

    alpha = 0.0042
    x = x0.copy()
    gd = [x.copy()]
    for _ in range(900):
        x = x - alpha * grad(x)
        gd.append(x.copy())
    gd = np.asarray(gd)

    xg = np.linspace(-1.7, 1.7, 260)
    yg = np.linspace(-0.25, 2.3, 260)
    xx, yy = np.meshgrid(xg, yg)
    zz = (1 - xx) ** 2 + scale * (yy - xx**2) ** 2
    return {
        "title": r"Rosenbrock valley, $f=(1-x)^2+6(y-x^2)^2$",
        "xlim": (-1.7, 1.7),
        "ylim": (-0.25, 2.3),
        "grid": (xx, yy, zz),
        "trajs": {"GF": gf, "AGF": agf, "GD": gd},
        "colors": {"GF": COLORS["gf"], "AGF": COLORS["agf"], "GD": COLORS["gd"]},
        "minimizer": (1.0, 1.0),
    }
