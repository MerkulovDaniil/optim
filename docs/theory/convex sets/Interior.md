---
layout: default
parent: Convex sets
grand_parent: Theory
title: Interior
nav_order: 4
---

# Interior

Внутренностью множества $$S$$ называется следующее множество: 

$$
\mathbf{int} (S) = \{\mathbf{x} \in S \mid \exists \varepsilon > 0, \; B(\mathbf{x}, \varepsilon) \subset S\}
$$

где $$B(\mathbf{x}, \varepsilon) = \mathbf{x} + \varepsilon B$$ - шар с центром в т. $$\mathbf{x}$$ и радиусом $$\varepsilon$$

# Relative interior
Относительной внутренностью множества $$S$$ называется следующее множество: 

$$
\mathbf{relint} (S) = \{\mathbf{x} \in S \mid \exists \varepsilon > 0, \; B(\mathbf{x}, \varepsilon) \cap \mathbf{aff} (S) \subseteq S\}
$$


![](../rel_int.svg)

Любое непустое выпуклое множество $$S \subseteq \mathbb{R}^n$$ имеет непустую относительную внутренность $$\mathbf{relint}(S)$$
