---
title: Uncategorized
parent: Exercises
order: 666
---

# Uncategorized{.unnumbered}

1. Show, that these conditions are equivalent:
    
    $$
     \|\nabla f(x) - \nabla f(z) \| \le L \|x-z\| 
    $$
    
    and
    
    $$
    f(z) \le f(x) + \nabla f(x)^T(z-x) + \frac L 2 \|z-x\|^2
    $$

1. We say that the function belongs to the class $f  \in C^{k,p}_L (Q)$ if it is $k$ times continuously differentiable on $Q$, and the $p$ derivative has a Lipschitz constant $L$. 

    $$
    \|\nabla^p f(x) - \nabla^p f(y)\| \leq L \|x-y\|, \qquad \forall x,y \in Q
    $$

    The most commonly used $C_L^{1,1}, C_L^{2,2}$ for $\mathbb{R}^n$. 
    Notice that:
    * $p \leq k$
    * If $q \geq k$, then $C_L^{q,p} \subseteq C_L^{k,p}$. The higher is the order of the derivative, the stronger is the limitation (fewer functions belong to the class).

    Prove that the function belongs to the class $C_L^{2,1}. \subseteq C_L^{1,1}$ if and only if $\forall x \in \mathbb{R}^n$:

    $$
    \|\nabla^2 f(x)\| \leq L
    $$

    Prove that the last condition can be rewritten in the form without loss of generality:

    $$
    -L I_n \preceq \nabla^2 f(x) \preceq L I_n
    $$

1. Show that for gradient descent with the following stepsize selection strategies:
    * constant step $h_k = \dfrac{1}{L}$
    * Dropping sequence $h_k = \dfrac{\alpha_k}{L}, \quad \alpha_k \to 0$.

    you can get the estimation of the function decrease at the iteration of the view:

    $$
    f(x_k) - f(x_{k+1}) \geq \dfrac{\omega}{L}\|\nabla f(x_k)\|^2
    $$

    $\omega > 0$ - some constant, $L$ - Lipschitz constant of the function gradient.