---
layout: default
title: Matrix calculus
parent: Excersises
nav_order: 0
---

# Matrix calculus

1. Найти $$\nabla f(x)$$, если $$f(x) = \|Ax\| $$
1. Найти $$\nabla f(x), f''(x)$$, если $$f(x) = \dfrac{-1}{1 + x^Tx}$$
1. Найти $$f'(X)$$, если $$f(X) = \det X$$  

    Примечание: здесь под $$f'(X)$$ подразумевается оценка фунции $$f(X)$$ первого порядка в смысле разложения в ряд Тейлора:
    $$
    f(X + \Delta X) \approx f(X) + \mathbf{tr}(f'(X)^T \Delta X)
    $$

1. Найти $$f''(X)$$, если $$f(X) = \log \det X$$  
   
    Примечание: здесь под $$f''(X)$$ подразумевается оценка фунции $$f(X)$$ второго порядка в смысле разложения в ряд Тейлора:
    $$
    f(X + \Delta X) \approx f(X) + \mathbf{tr}(f'(X)^T \Delta X) + \frac{1}{2}\mathbf{tr}(\Delta X^T f''(X) \Delta X)
    $$

1. Найти градиент и гессиан функции $$f : \mathbb{R}^n \to \mathbb{R}$$, если:

    $$
    f(x) = \log \sum\limits_{i=1}^m \exp (a_i^Tx + b_i), \;\;\;\; a_1, \ldots, a_m \in \mathbb{R}^n; \;\;\;  b_1, \ldots, b_m  \in \mathbb{R}
    $$

1. Описать концептуальную схему решения задач на векторное\матричное дифференцирование
1. Что такое градиент, гессиан, якобиан? Их связь?
1. Посчитать: $$\dfrac{\partial }{\partial X} \sum \text{eig}(X), \;\;\dfrac{\partial }{\partial X} \prod \text{eig}(X), \;\;\dfrac{\partial }{\partial X}\text{tr}(X), \;\; \dfrac{\partial }{\partial X} \text{det}(X)$$
1. Посчитать матричную производную нормы Фробениуса: $$\dfrac{\partial}{\partial X}\|X\|_F^2$$
1. Посчитать градиент логистической регрессии $$\nabla_\theta L$$ в случае двух классов ($$K = 2$$) $$n$$ - мерных объектов:
	$$
	h_\theta(x) =
	\begin{bmatrix}
	P(y = 1 | x; \theta) \\
	P(y = 2 | x; \theta) \\
	\vdots \\
	P(y = K | x; \theta)
	\end{bmatrix}
	=
	\frac{1}{ \sum_{j=1}^{K}{\exp(\theta^{(j)\top} x) }}
	\begin{bmatrix}
	\exp(\theta^{(1)\top} x ) \\
	\exp(\theta^{(2)\top} x ) \\
	\vdots \\
	\exp(\theta^{(K)\top} x ) \\
	\end{bmatrix}
	$$

	$$
	L(\theta) = - \left[ \sum_{i=1}^n  (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) + y^{(i)} \log h_\theta(x^{(i)}) \right]
	$$

1. Посчитать $$\nabla f(X)$$, если $$f(X) = \text{tr } AX$$
1. Посчитать $$\nabla f(X)$$, if $$f(X) = \langle S, X\rangle - \log \det X$$
1. Посчитать $$\nabla f(X)$$, if $$f(X) = \ln \langle Ax, x\rangle, A \in \mathbb{S^n_{++}}$$
1. Посчитать $$\nabla f(x), f''(x)$$, if $$f(x) = \frac{1}{3}\|x\|_2^3$$
1. Посчитать $$\nabla f(x), f''(x)$$, if $$f(x) = \ln (1 + \exp{\langle a,x\rangle})$$
1. Посчитать $$\nabla f(X)$$, если $$f(X) = \| AX - B\|_F, X \in \mathbb{R}^{k \times n}, A \in \mathbb{R}^{m \times k}, B \in \mathbb{R}^{m \times n}$$