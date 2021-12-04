---
layout: default
has_children: true
title: Methods
nav_order: 3
---

# General formulation

$$
\begin{split}
& \min_{x \in \mathbb{R}^n} f(x)\\
\text{s.t. }  g_i(x) \leq& 0, \; i = 1,\ldots,m\\
 h_j(x) =& 0, \; j = 1,\ldots,k\\
\end{split}
$$

Some necessary or/and sufficient conditions are known (See {% include link.html title='Optimality conditions. KKT'%} and {% include link.html title='Convex optimization problem' %})
* In fact, there might be very challenging to recognize the convenient form of optimization problem.
* Analytical solution of KKT could be inviable.

## Iterative methods
Typically, the methods generate an infinite sequence of approximate solutions

$$
\{x_t\},
$$

which for a finite number of steps (or better - time) converges to an optimal (at least one of the optimal) solution  $$x_*$$.

![](../iterative.svg)

```python
def GeneralScheme(x, epsilon):
    while not StopCriterion(x, epsilon):
        OracleResponse = RequestOracle(x)
        x = NextPoint(x, OracleResponse)
    return x
```

## Oracle conception

![](../oracle.svg)

## Complexity

# Challenges

## Unsolvability
In general, **optimization problems are unsolvable.**  ¯\\_(ツ)_/¯

Consider the following simple optimization problem of a function over unit cube:

$$
\begin{split}
& \min_{x \in \mathbb{R}^n} f(x)\\
\text{s.t. } &  x \in \mathbb{B}^n
\end{split}
$$

We assume, that the objective function $$f (\cdot) : \mathbb{R}^n \to \mathbb{R}$$ is Lipschitz continuous on
$$\mathbb{B}^n$$:

$$
| f (x) − f (y) | \leq L \| x − y \|_{\infty} \forall x,y \in \mathbb{B}^n,
$$

with some constant $$L$$ (Lipschitz constant). Here $$\mathbb{B}^n$$ - the $$n$$-dimensional unit cube 

$$
\mathbb{B}^n = \{x \in \mathbb{R}^n \mid 0 \leq x_i \leq 1, i = 1, \ldots, n\}
$$ 

Our goal is to find such $$\tilde{x}: \vert f(\tilde{x}) - f^*\vert \leq \varepsilon$$ for some positive $$\varepsilon$$. Here $$f^*$$ is the global minima of the problem. Uniform grid with $$p$$ points on each dimension guarantees at least this quality:

$$
\| \tilde{x} − x_* \|_{\infty} \leq \frac{1}{2p},
$$

which means, that

$$
|f (\tilde{x}) − f (x_*)| \leq \frac{L}{2p}
$$

Our goal is to find the $$p$$ for some $$\varepsilon$$. So, we need to sample $$ \left(\frac{L}{2 \varepsilon}\right)^n$$ points, since we need to measure function in $$p^n$$ points. Doesn't look scary, but if we'll take $$L = 2, n = 11, \varepsilon = 0.01$$, computations on the modern personal computers will take 31,250,000 years.

## Stopping rules
* Argument closeness: 

$$
\| x_k - x_*  \|_2 < \varepsilon
$$ 

* Function value closeness: 

$$
\| f_k - f^* \|_2 < \varepsilon
$$ 

* Closeness to a critical point

$$
\| f'(x_k) \|_2 < \varepsilon
$$

But $$x_*$$ and $$f^* = f(x_*)$$ are unknown!

Sometimes, we can use the trick:

$$
\|x_{k+1} - x_k \| = \|x_{k+1} - x_k + x_* - x_* \| \leq \|x_{k+1} - x_* \| + \| x_k - x_* \| \leq 2\varepsilon
$$

**Note**: it's better to use relative changing of these values, i.e. $$\dfrac{\|x_{k+1} - x_k \|_2}{\| x_k \|_2}$$.


## Local nature of the methods

![](../globallocal.png)

# Problem classifications

# Methods classifications

# Speed of convergence
* Sublinear

	$$
	\| x_{k+1} - x_* \|_2 \leq C k^{\alpha},
	$$

	where $$\alpha < 0$$ and $$ 0 < C < \infty$$

* Linear

	$$
	\| x_{k+1} - x_* \|_2 \leq Cq^k \qquad\text{or} \qquad \| x_{k+1} - x_* \|_2 \leq q\| x_k - x_* \|_2,
	$$

	where $$q \in (0, 1)$$ and $$ 0 < C < \infty$$

* Superlinear 

	$$
	\| x_{k+1} - x_* \|_2 \leq Cq^{k^2} \qquad \text{or} \qquad \| x_{k+1} - x_* \|_2 \leq C_k\| x_k - x_* \|_2,
	$$

	where $$q \in (0, 1)$$ or $$ 0 < C_k < \infty$$, $$C_k \to 0$$

* Quadratic

	$$
	\| x_{k+1} - x_* \|_2 \leq C q^{2^k} \qquad \text{or} \qquad \| x_{k+1} - x_* \|_2 \leq C\| x_k - x_* \|^2_2,
	$$

	where $$q \in (0, 1)$$ and $$ 0 < C < \infty$$

![](../convergence.svg)

## Root test

Пусть $$\{r_k\}_{k=m}^\infty$$ — последовательность неотрицательных чисел,
сходящаяся к нулю, и пусть 

$$ 
\alpha = \lim_{k \to \infty} \sup_k \; r_k ^{1/k}
$$

* Если $$0 \leq \alpha \lt 1$$, то $$\{r_k\}_{k=m}^\infty$$ имеет линейную сходимость с константной $$\alpha$$. 
* В частности, если $$\alpha = 0$$, то $$\{r_k\}_{k=m}^\infty$$ имеет сверхлинейную сходимость.
* Если $$\alpha = 1$$, то $$\{r_k\}_{k=m}^\infty$$ имеет сублинейную сходимость. 
* Случай $$\alpha \gt 1$$ невозможен.

## Ratio test

Пусть $$\{r_k\}_{k=m}^\infty$$ — последовательность строго положительных чисел, сходящаяся к нулю. Пусть

$$
\alpha = \lim_{k \to \infty} \dfrac{r_{k+1}}{r_k}
$$

* Если существует $$\alpha$$ и при этом $$0 \leq \alpha \lt  1$$, то $$\{r_k\}_{k=m}^\infty$$ имеет линейную сходимость с константой $$\alpha$$
* В частности, если $$\alpha = 0$$, то $$\{r_k\}_{k=m}^\infty$$ имеет сверхлинейную сходимость
* Если $$\alpha$$ не существует, но при этом $$q = \lim\limits_{k \to \infty} \sup_k \dfrac{r_{k+1}}{r_k} \lt  1$$, то $$\{r_k\}_{k=m}^\infty$$ имеет линейную сходимость с константой, не превосходящей $$q$$. 
* Если $$ \lim\limits_{k \to \infty} \inf_k \dfrac{r_{k+1}}{r_k} =1$$, то $$\{r_k\}_{k=m}^\infty$$ имеет сублинейную сходимость. 
* Ситуация $$ \lim\limits_{k \to \infty} \inf_k \dfrac{r_{k+1}}{r_k} \gt 1$$ невозможна. 
* Во всех остальных случаях (т. е. когда $$ \lim\limits_{k \to \infty} \inf_k \dfrac{r_{k+1}}{r_k} \lt  1 \leq  \lim\limits_{k \to \infty} \sup_k \dfrac{r_{k+1}}{r_k}$$) нельзя утверждать что-либо конкретное о скорости сходимости $$\{r_k\}_{k=m}^\infty$$.

# References
* Code for convergence plots - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Convergence.ipynb)
* [CMC seminars (ru)](http://www.machinelearning.ru/wiki/images/9/9a/MOMO18_Extra1.pdf)
