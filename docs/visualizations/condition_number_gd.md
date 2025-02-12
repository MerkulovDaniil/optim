---
title: "Condition number and gradient descent"
---

Suppose, we have a strongly convex quadratic function minimization problem solved by the gradient descent method:
$$
f(x) = \frac{1}{2} x^T A x - b^T x \qquad x^{k+1} = x^k - \alpha_k \nabla f(x^k).
$$

The gradient descent method with the learning rate $\alpha_k = \frac{2}{\mu + L}$ converges to the optimal solution $x^*$ with the following guarantee:
$$
\|x^{k+1} - x^*\|_2 = \left( \frac{\varkappa-1}{\varkappa+1}\right)^k \|x^0 - x^*\|_2 \qquad f(x^{k+1}) - f(x^*) = \left( \frac{\varkappa-1}{\varkappa+1}\right)^{2k} \left(f(x^0) - f(x^*)\right),
$$
where $\varkappa$ is the condition number of the Hessian matrix $A$.


:::{.video}
condition_number_gd.mp4
:::


[Code](condition_number_gd.py)
