---
layout: default
title: Successive parabolic interpolation
parent: Line search
grand_parent: Methods
---
# Idea
Sampling 3 points of a function determines unique parabola. Using this information we will go directly to its minimum. Suppose, we have 3 points $$x_1 < x_2 < x_3$$ such that line segment $$[x_1, x_3]$$ contains minimum of a function $$f(x)$$. Then, we need to solve the following system of equations:

$$
ax_i^2 + bx_i + c = f_i = f(x_i), i = 1,2,3 
$$

Note, that this system is linear, since we need to solve it on $$a,b,c$$. Minimum of this parabola will be calculated as:

$$
u = -\dfrac{b}{2a} = x_2 - \dfrac{(x_2 - x_1)^2(f_2 - f_3) - (x_2 - x_3)^2(f_2 - f_1)}{2\left[ (x_2 - x_1)(f_2 - f_3) - (x_2 - x_3)(f_2 - f_1)\right]}
$$

Note, that if $$f_2 < f_1, f_2 < f_3$$, than $$u$$ will lie in $$[x_1, x_3]$$

# Algorithm
```python
def parabola_search(f, x1, x2, x3, epsilon):
    f1, f2, f3 = f(x1), f(x2), f(x3)
    while x3 - x1 > epsilon:
        u = x2 - ((x2 - x1)**2*(f2 - f3) - (x2 - x3)**2*(f2 - f1))/(2*((x2 - x1)*(f2 - f3) - (x2 - x3)*(f2 - f1)))
        fu = f(u)

        if x2 <= u:
            if f2 <= fu:
                x1, x2, x3 = x1, x2, u
                f1, f2, f3 = f1, f2, fu
            else:
                x1, x2, x3 = x2, u, x3
                f1, f2, f3 = f2, fu, f3
        else:
            if fu <= f2:
                x1, x2, x3 = x1, u, x2
                f1, f2, f3 = f1, fu, f2
            else:
                x1, x2, x3 = u, x2, x3
                f1, f2, f3 = fu, f2, f3
    return (x1 + x3) / 2
```
# Bounds

The convergence of this method is superlinear, but local, which means, that you can take profit from using this method only near some neighbour of optimum.
