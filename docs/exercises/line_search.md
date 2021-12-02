---
layout: default
title: Line search
parent: Exercises
nav_order: 13
---

1. Which function is called unimodal?
1. Derive the convergence speed for a dichotomy method for a unimodal function. What type of convergence does this method have?
1. Consider the function $$f(x) = (x + \sin x) e^x, \;\;\; x \in [-20, 0]$$. 
    ![](../Unimodal.svg)
    Consider the following modification of solution localization method, in which the interval $$[a,b]$$ is divided into $$2$$ parts in a fixed proportion of $$t: x_t = a + t*(b-a)$$ (maximum twice at iteration - as in the dichotomy method). Experiment with different values of $$t \in [0,1]$$ and plot the dependence of $$N (t)$$ - the number of iterations needed to achieve $$\varepsilon$$ - accuracy from the $$t$$ parameter. Consider $$\varepsilon = 10^{-7}$$. Note that with $$t = 0.5$$ this method is exactly the same as the dichotomy method.
1. Describe the idea of successive parabolic interpolation. What type of convergence does this method have?
1. Write down Armijo–Goldstein condition. 
