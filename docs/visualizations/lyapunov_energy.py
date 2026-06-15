"""Lyapunov energy for the accelerated gradient flow (convex case)."""
from __future__ import annotations

from lyapunov_common import render

TITLE = "Lyapunov energy of the accelerated flow (convex)"
DESCRIPTION = r"""
The accelerated gradient flow $\ddot X + \tfrac{3}{t}\dot X + \nabla f(X)=0$
reaches the $\mathcal{O}(1/t^2)$ rate. The proof uses the Lyapunov energy
$$
E(t)=t^2\bigl(f(X(t))-f^*\bigr)+2\Bigl\|X(t)-x^*+\tfrac{t}{2}\dot X(t)\Bigr\|^2,
$$
which is non-increasing along the trajectory. Since $E(t)\le E(0)=2\|x_0-x^*\|^2$,
the first term gives $f(X(t))-f^*\le 2\|x_0-x^*\|^2/t^2$. The objective
**oscillates** under the curve while the energy decreases monotonically.
"""

if __name__ == "__main__":
    render("lyapunov_energy", TITLE, DESCRIPTION, "convex")
