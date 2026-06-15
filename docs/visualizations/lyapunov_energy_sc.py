"""Lyapunov energy for the accelerated gradient flow (strongly convex case)."""
from __future__ import annotations

from lyapunov_common import render

TITLE = "Lyapunov energy of the accelerated flow (strongly convex)"
DESCRIPTION = r"""
For $\mu$-strongly convex $f$ the right flow has **constant** damping
$\ddot X + 2\sqrt{\mu}\,\dot X + \nabla f(X)=0$ and converges **linearly**. The
Lyapunov energy
$$
E(t)=\bigl(f(X(t))-f^*\bigr)+\tfrac12\bigl\|\dot X(t)+\sqrt{\mu}\,(X(t)-x^*)\bigr\|^2
$$
satisfies $\dot E\le-\sqrt{\mu}\,E$, hence $E(t)\le E(0)e^{-\sqrt{\mu}\,t}$ and
$f(X(t))-f^*\le E(0)e^{-\sqrt{\mu}\,t}$ — a straight line on the semi-log plot.
"""

if __name__ == "__main__":
    render("lyapunov_energy_sc", TITLE, DESCRIPTION, "sc")
