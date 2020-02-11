---
layout: default
title: Golden search
parent: Line search
grand_parent: Methods
---

# Idea
The idea is quite similar to the dichotomy method. There are two golden points on the line segment (left and right) and the insightful idea is, that on the next iteration one of the points wtill remains the golden point.

![](../golden_search.svg)

# Algorithm
```python
def golden_search(f, a, b, epsilon):
    tau = (sqrt(5) + 1) / 2
    y = a + (b - a) / tau**2
    z = a + (b - a) / tau
    while b - a > epsilon:
        if f(y) <= f(z):
            b = z
            z = y
            y = a + (b - a) / tau**2
        else:
            a = y
            y = z
            z = a + (b - a) / tau
    return (a + b) / 2
```
# Bounds

$$
|x_{k+1} - x_*| \leq b_{k+1} - a_{k+1} = \left( \frac{1}{\tau} \right)^{N-1} (b - a) \approx 0.618^k(b-a),
$$

where $\tau = \frac{\sqrt{5} + 1}{2}$.

* The geometric progression constant **more** than the dichotomy method - $$0.618$$ worse than $$0.5$$
* The number of function calls ** is less** than for the dichotomy method - $$0.707$$ worse than $$0.618$$ - (for each iteration of the dichotomy method, except for the first one, the function is calculated no more than 2 times, and for the gold method - no more than one)
