---
layout: default
title: Simulated Annealing
parent: Zero order methods
grand_parent: Methods
nav_order: 2
---

# Fofrmulation of a Problem

We need to optimize the global optimum of a given function on some space using only the values of the function in some points on the space.

$$\min_{x \in X} F(x) = F(x^*)$$

Simulated Annealing is a probabilistic technique for approximating the global optimum of a given function.

# Algorithm

## Main Idea

The name and inspiration come from annealing in metallurgy, a technique involving heating and controlled cooling of a material to increase the size of its crystals and reduce their defects. Both are attributes of the material that depend on its thermodynamic free energy. Heating and cooling the material affects both the temperature and the thermodynamic free energy. The simulation of annealing can be used to find an approximation of a global minimum for a function with many variables.

## Steps of the algorithm

1️⃣ Let $$ k = 0 $$ - current iteration, $$T = T_k$$ - initial temperature.

2️⃣ Let $$x_k \in X$$ - some random point from our space

3️⃣ Let decrease the temperature by following rule $$T_{k+1} = \alpha T_k$$ where $$ 0 < \alpha < 1$$ - some constant that often is closer to 1

4️⃣ Let $$x_{k+1} = g(x_k)$$ - the next point which was obtained from previous one by some random rule. It is usually assumed that this rule works so that each subsequent approximation should not differ very much.

5️⃣ Calculate $$\Delta E = E(x_{k+1}) - E(x_{k})$$, where $$E(x)$$ - the function that determines the energy of the system at this point. It is supposed that energy has the minimum in desired value $$x^*$$.

6️⃣ If $$\Delta E < 0$$ then the approximation found is better than it was. So accept $$x_{k+1}$$ as new started point at the next step and go to the step 3️⃣.

7️⃣ If $$\Delta E < 0$$, then we accept $$x_{k+1}$$ with the probability of $$P(\Delta E) = \exp^{-\Delta E / T_k}$$. If we don't accept $$x_{k+1}$$, then we let $$k = k+ 1$$. Go to the step 3️⃣.

The algorithm can stop working according to various criteria, for example, achieving an optimal state or lowering the temperature below a predetermined level $$T_{min}$$.

## Theoretical facts

As it mentioned in [Simulated annealing: a proof of convergence](https://ieeexplore.ieee.org/document/295910) the algorithm converges almost surely to a global maximum.

## Illustration

A gif from [Wikipedia](https://en.wikipedia.org/wiki/Markdown):

![](../sa_wiki.gif)

# Application Example

In our example we solve the N queens puzzle - the problem of placing N chess queens on an N×N chessboard so that no two queens threaten each other.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)](https://colab.research.google.com/drive/1NTBSgC_fUuqt9YxN68cUq00tLcjfw-vy)

## The Problem

Let $$E(x)$$ - the number of intersections, where $$x$$ - the array of placement queens at the field (the number in array means the column, the index of the number means the row).

**_The problem is_** to find $$x^*$$ where $$E$$ reach the global minimum, that is predefined and equals to 0 (no two queens threaten each other).

In this code $$x_0 = [0,1,2,...,N]$$ that means all queens are placed at the board's diagonal . So at the begining $$E = N(N-1)$$, because every queen intersects others.

$$\alpha = 0.95$$

## Results

Results of applying this algorithm to the problem below:

![](../sa_result.svg)
