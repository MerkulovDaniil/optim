---
parent: Theory
title: Rates of convergence
---

## Speed of convergence
In order to compare perfomance of algorithms we need to define a terminology for different types of convergence.
Let $r_k = \{\|x_k - x^*\|_2\}$ be a sequence in $\mathbb{R}^n$ that converges to zero.

### Linear convergence

We can define the *linear* convergence in a two different forms:

$$
\| x_{k+1} - x^* \|_2 \leq Cq^k \quad\text{or} \quad \| x_{k+1} - x^* \|_2 \leq q\| x_k - x^* \|_2,
$$

for all sufficiently large $k$. Here $q \in (0, 1)$ and $0 < C < \infty$. This means that the distance to the solution $x^*$ decreases at each iteration by at least a constant factor bounded away from $1$. Note, that sometimes this type of convergence is also called *exponential* or *geometric*. We call the $q$ the convergence rate.

:::{.callout-question}
Suppose, you have two sequences with linear convergence rates $q_1 = 0.1$ and $q_2 = 0.7$, which one is faster?
:::

:::{.callout-example}
Let us have the following sequence:

$$
r_k = \dfrac{1}{3^k}
$$

One can immediately conclude, that we have a linear convergence with parameters $q = \dfrac{1}{3}$ and $C = 0$.
:::

:::{.callout-question}
Let us have the following sequence:

$$
r_k = \dfrac{4}{3^k}
$$

Will this sequence be convergent? What is the convergence rate?
:::

### Sublinear convergence

If the sequence $r_k$ converges to zero, but does not have linear convergence, the convergence is said to be sublinear. Sometimes we can considet the following class of sublinear convergence:

$$
\| x_{k+1} - x^* \|_2 \leq C k^{q},
$$

where $q < 0$ and $0 < C < \infty$. Note, that sublinear convergence means, that the sequence is converging slower, than any geometric progression.

### Superlinear convergence

The convergence is said to be *superlinear* if:

$$
\| x_{k+1} - x^* \|_2 \leq Cq^{k^2} \qquad \text{or} \qquad \| x_{k+1} - x^* \|_2 \leq C_k\| x_k - x^* \|_2,
$$

where $q \in (0, 1)$ or $0 < C_k < \infty$, $C_k \to 0$. Note, that superlinear convergence is also linear convergence (one can even say, that it is linear convergence with $q=0$).

### Quadratic convergence

$$
\| x_{k+1} - x^* \|_2 \leq C q^{2^k} \qquad \text{or} \qquad \| x_{k+1} - x^* \|_2 \leq C\| x_k - x^* \|^2_2,
$$

where $q \in (0, 1)$ and $0 < C < \infty$.

![Difference between the convergence speed](convergence.svg)

Quasi-Newton methods for unconstrained optimization typically converge superlinearly, whereas Newton’s method converges quadratically under appropriate assumptions. In contrast, steepest descent algorithms converge only at a linear rate, and when the problem is ill-conditioned the convergence constant $q$ is close to $1$.

## How to determine convergence type

### Root test

Let $\{r_k\}_{k=m}^\infty$ be a sequence of non-negative numbers,
converging to zero, and let 

$$ 
q = \lim_{k \to \infty} \sup_k \; r_k ^{1/k}
$$

* If $0 \leq q \lt 1$, then $\{r_k\}_{k=m}^\infty$ has linear convergence with constant $q$. 
* In particular, if $q = 0$, then $\{r_k\}_{k=m}^\infty$ has superlinear convergence.
* If $q = 1$, then $\{r_k\}_{k=m}^\infty$ has sublinear convergence.
* The case $q \gt 1$ is impossible.

### Ratio test

Let $\{r_k\}_{k=m}^\infty$ be a sequence of strictly positive numbers converging to zero. Let

$$
q = \lim_{k \to \infty} \dfrac{r_{k+1}}{r_k}
$$

* If there exists $q$ and $0 \leq q \lt  1$, then $\{r_k\}_{k=m}^\infty$ has linear convergence with constant $q$.
* In particular, if $q = 0$, then $\{r_k\}_{k=m}^\infty$ has superlinear convergence.
* If $q$ does not exist, but $q = \lim\limits_{k \to \infty} \sup_k \dfrac{r_{k+1}}{r_k} \lt  1$, then $\{r_k\}_{k=m}^\infty$ has linear convergence with a constant not exceeding $q$. 
* If $\lim\limits_{k \to \infty} \inf_k \dfrac{r_{k+1}}{r_k} =1$, then $\{r_k\}_{k=m}^\infty$ has sublinear convergence. 
* The case $\lim\limits_{k \to \infty} \inf_k \dfrac{r_{k+1}}{r_k} \gt 1$ is impossible. 
* In all other cases (i.e., when $\lim\limits_{k \to \infty} \inf_k \dfrac{r_{k+1}}{r_k} \lt  1 \leq  \lim\limits_{k \to \infty} \sup_k \dfrac{r_{k+1}}{r_k}$) we cannot claim anything concrete about the convergence rate $\{r_k\}_{k=m}^\infty$.

:::{.callout-example}
Let us have the following sequence:

$$
r_k = \dfrac{1}{k}
$$

Determine the convergence
:::

:::{.callout-example}
Let us have the following sequence:

$$
r_k = \dfrac{1}{k^2}
$$

Determine the convergence
:::

:::{.callout-example}
Let us have the following sequence:

$$
r_k = \dfrac{1}{k^q}, q > 1
$$

Determine the convergence
:::

:::{.callout-example}

## Try to use root test here
Let us have the following sequence:

$$
r_k = \dfrac{1}{k^k}
$$

Determine the convergence
:::

## References
* Code for convergence plots - [Open In Colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Convergence.ipynb)
* [CMC seminars (ru)](http://www.machinelearning.ru/wiki/images/9/9a/MOMO18_Extra1.pdf)
* Numerical Optimization by J.Nocedal and S.J.Wright