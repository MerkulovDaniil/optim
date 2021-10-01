---
layout: default
title: Matrix calculus
parent: Exercises
nav_order: 0
---

# Matrix calculus

1. Find the derivatives of $$f(x) = Ax, \quad \nabla_x f(x) = ?, \nabla_A f(x) = ?$$
1. Find $$\nabla f(x)$$, if $$f(x) = c^Tx$$.
1. Find $$\nabla f(x)$$, if $$f(x) = \dfrac{1}{2}x^TAx + b^Tx + c$$.
1. Find $$\nabla f(x), f''(x)$$, if $$f(x) = -e^{-x^Tx}$$.
1. Find the gradient $$\nabla f(x)$$ and hessian $$f''(x)$$, if $$f(x) = \dfrac{1}{2} \|Ax - b\|^2_2$$.
1. Find $$\nabla f(x)$$, if $$f(x) = \|x\|_2 , x \in \mathbb{R}^p \setminus \{0\}$$.
1. Find $$\nabla f(x)$$, if $$f(x) = \|Ax\|_2 , x \in \mathbb{R}^p \setminus \{0\}$$.
1. Find $$\nabla f(x), f''(x)$$, if $$f(x) = \dfrac{-1}{1 + x^\top x}$$.
1. Calculate $$df(x)$$ and $$\nabla f(x)$$ for the function $$f(x) = \log(x^{\top}\mathrm{A}x)$$.
1. Find $$f'(X)$$, if $$f(X) = \det X$$  

    Note: here under $$f'(X)$$ assumes first order approximation of $$f(X)$$ using Taylor series:
    $$
    f(X + \Delta X) \approx f(X) + \mathbf{tr}(f'(X)^\top \Delta X)
    $$

1. Find $$f''(X)$$, if $$f(X) = \log \det X$$  
   
    Note: here under $$f''(X)$$ assumes second order approximation of $$f(X)$$ using Taylor series:
    $$
    f(X + \Delta X) \approx f(X) + \mathbf{tr}(f'(X)^\top \Delta X) + \frac{1}{2}\mathbf{tr}(\Delta X^\top f''(X) \Delta X)
    $$

1. Find gradient and hessian of $$f : \mathbb{R}^n \to \mathbb{R}$$, if:

    $$
    f(x) = \log \sum\limits_{i=1}^m \exp (a_i^\top x + b_i), \;\;\;\; a_1, \ldots, a_m \in \mathbb{R}^n; \;\;\;  b_1, \ldots, b_m  \in \mathbb{R}
    $$

1. What is the gradient, Jacobian, Hessian? Is there any connection between those three definitions?
1. Calculate: $$\dfrac{\partial }{\partial X} \sum \text{eig}(X), \;\;\dfrac{\partial }{\partial X} \prod \text{eig}(X), \;\;\dfrac{\partial }{\partial X}\text{tr}(X), \;\; \dfrac{\partial }{\partial X} \text{det}(X)$$
1. Calculate the Frobenious norm derivative: $$\dfrac{\partial}{\partial X}\|X\|_F^2$$
1. Calculate the gradient of the softmax regression $$\nabla_\theta L$$ in binary case ($$K = 2$$) $$n$$ - dimensional objects:
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

1. Find $$\nabla f(X)$$, if $$f(X) = \text{tr } AX$$
1. Find $$\nabla f(X)$$, if $$f(X) = \langle S, X\rangle - \log \det X$$
1. Find $$\nabla f(X)$$, if $$f(X) = \ln \langle Ax, x\rangle, A \in \mathbb{S^n_{++}}$$
1. Find the gradient $$\nabla f(x)$$ and hessian $$f''(x)$$, if 
    
    $$
    f(x) = \ln \left( 1 + \exp\langle a,x\rangle\right)
    $$

1. Find the gradient $$\nabla f(x)$$ and hessian $$f''(x)$$, if $$f(x) = \frac{1}{3}\|x\|_2^3$$
1. Calculate $$\nabla f(X)$$, if $$f(X) = \| AX - B\|_F, X \in \mathbb{R}^{k \times n}, A \in \mathbb{R}^{m \times k}, B \in \mathbb{R}^{m \times n}$$
1. Calculate the derivatives of the loss function with respect to parameters $$\frac{\partial L}{\partial W}, \frac{\partial L}{\partial b}$$ for the single object $$x_i$$ (or, $$n = 1$$)
![](../simple_learning.svg)
1. Find the gradient $$\nabla f(x)$$ and hessian $$f''(x)$$, if $$f(x) = \langle x, x\rangle^{\langle x, x\rangle}, x \in \mathbb{R}^p \setminus \{0\}$$
1. Find the gradient $$\nabla f(x)$$ and hessian $$f''(x)$$, if $$f(x) = \frac{\langle Ax, x\rangle}{\|x\|_2^2}, x \in \mathbb{R}^p \setminus \{0\}, A \in \mathbb{S}^n$$
1. Find the gradient $$\nabla f(x)$$ and hessian $$f''(x)$$, if $$f(x) = \frac{1}{2}\|A - xx^\top\|^2_F, A \in \mathbb{S}^n$$
1. Find the gradient $$\nabla f(x)$$ and hessian $$f''(x)$$, if $$f(x) = \|xx^\top\|_2$$
1. Find the gradient $$\nabla f(x)$$ and hessian $$f''(x)$$, if $$f(x) = \frac1n \sum\limits_{i=1}^n \log \left( 1 + \exp(a_i^{\top}x)  \right) + \frac{\mu}{2}\|x\|_2^2, \; a_i \in \mathbb R^n, \; \mu>0$$.
1. Match functions with their gradients:
    1. \$$f(\mathrm{X}) = \mathrm{Tr}\mathrm{X}$$
    1. \$$f(\mathrm{X}) = \mathrm{Tr}\mathrm{X}^{-1}$$
    1. \$$f(\mathrm{X}) = \det \mathrm{X}$$
    1. \$$f(\mathrm{X}) = \ln \det \mathrm{X}$$

    a. \$$\nabla f(\mathrm{X}) = \mathrm{X}^{-1}$$
    b. \$$\nabla f(\mathrm{X}) = \mathrm{I}$$
    c. \$$\nabla f(\mathrm{X}) = \det (\mathrm{X})\cdot (\mathrm{X}^{-1})^{\top}$$
    d. \$$\nabla f(\mathrm{X}) = -\left(\mathrm{X}^{-2}\right)^{\top}$$
1. Calculate the first and the second derivative of the following function $$f : S \to \mathbb{R}$$
	$$
	f(t) = \text{det}(A − tI_n),
	$$
	where $$A \in \mathbb{R}^{n \times n}, S := \{t \in \mathbb{R} : \text{det}(A − tI_n) \neq 0\}	$$.	
1. Find the gradient $$\nabla f(x)$$, if $$f(x) = \text{tr}\left( AX^2BX^{-\top} \right)$$.
