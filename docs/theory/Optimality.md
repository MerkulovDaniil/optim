---
layout: default
parent: Theory
title: Optimality conditions. KKT
nav_order: 8
---

# Background
## Extreme value (Weierstrass) theorem
Пусть $$S \subset \mathbb{R}^n$$ - компактное множество и пусть $$f(x)$$ непрерывная функция на $$S$$. 
Тогда точка глобального минимума функции $$f (x)$$ на $$S$$ существует.

![](../goodnews.png)

## Lagrange multipliers
Consider simple yet practical case of equality constraints:

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
\text{s.t. } & h_i(x) = 0, i = 1, \ldots, m
\end{split}
$$

The basic idea of Lagrange method implies switch from conditional to unconditional optimization through increasing the dimensionality of the problem:

$$
L(x, \lambda) = f(x) + \sum\limits_{i=1}^m \lambda_i h_i(x) \to \min\limits_{x \in \mathbb{R}^n, \lambda \in \mathbb{R}^m} \\
$$

# General formulations and conditions


$$
f(x) \to \min\limits_{x \in S}
$$

Будем говорить, что задача имеет решение, если множество таких $x^* \in S$, что в них достигается минимум или инфимум данной функции **не пусто**


## Unconstrained optimization
### General case
Let $$f(x): \mathbb{R}^n \to \mathbb{R}$$ be a twice differentiable function.

$$
\tag{UP}
f(x) \to \min\limits_{x \in \mathbb{R}^n}
$$

If $$x^*$$ - is a local minimum of $$f(x)$$, then:

$$
\tag{UP:Necessary}
\nabla f(x^*) = 0
$$

If $$f(x)$$ at some point $$x^*$$ satisfies the following conditions:

$$
\tag{UP:Sufficient}
H_f(x^*) = \nabla^2 f(x^*) \succeq (\preceq) 0,
$$

then (if necessary condition is also satisfied) $$x^*$$ is a local minimum(maximum) of $$f(x)$$.

### Convex case
It should be mentioned, that in **convex** case (i.e., $$f(x)$$ is convex) necessary condition becomes sufficient. Moreover, we can generalize this result on the class of non-differentiable convex functions. 

Let $$f(x): \mathbb{R}^n \to \mathbb{R}$$ - convex function, then the point $$x^*$$ is the solution of $$\text{(UP)}$$ if and only if:

$$
0_n \in \partial f(x^*)
$$

One more important result for convex constrained case sounds as follows. If $$f(x): S \to \mathbb{R}$$ - convex function defined on the convex set $$S$$, then:
* Any local minima is the global one.
* The set of the local minimizers $$S^*$$ is convex.
* If $$f(x)$$ - strongly convex function, then $$S^*$$ contains only one single point $$S^* = x^*$$.

## Optimization with equality conditions
### Intuition
Things are pretty simple and intuitive in unconstrained problem. In this section we will add one equality constraint, i.e.

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
\text{s.t. } & h(x) = 0
\end{split}
$$

We will try to illustrate approach to solve this problem through the simple example with $$f(x) = x_1 + x_2$$ and $$h(x) = x_1^2 + x_2^2 - 2$$

![](../kkt_images/KKT_p009.svg)

![](../kkt_images/KKT_p010.svg)

![](../kkt_images/KKT_p011.svg)

![](../kkt_images/KKT_p012.svg)

![](../kkt_images/KKT_p013.svg)

![](../kkt_images/KKT_p014.svg)

![](../kkt_images/KKT_p015.svg)

![](../kkt_images/KKT_p016.svg)

![](../kkt_images/KKT_p017.svg)

Итого, имеем: чтобы двигаясь из $$x_F$$ по бюджетному множеству в сторону уменьшения функции, нам необходимо гарантировать два условия:

$$
\langle \delta x, \nabla h(x_F) \rangle = 0
$$

$$
\langle \delta x, - \nabla f(x_F) \rangle > 0
$$

Пусть в процессе такого движения мы пришли в точку, где 
$$
\nabla f(x) = \lambda \nabla h(x)
$$

$$
\langle  \delta x, - \nabla f(x)\rangle = -\langle  \delta x, \lambda\nabla h(x)\rangle = 0  
$$

Тогда мы пришли в точку бюджетного множества, двигаясь из которой не получится уменьшить нашу функцию. Это и есть локальный минимум в ограниченной задаче:)

![](../kkt_images/KKT_p021.svg)

Так давайте определим функцию Лагранжа (исключительно для удобства):

$$
L(x, \lambda) = f(x) + \lambda h(x)
$$

Тогда точка $$x^*$$ - локальный минимум описанной выше задачи, тогда и только тогда, когда:

$$
\begin{split}
& \nabla_x L(x^*, \lambda^*) = 0 \text{ то, что мы написали выше}\\
& \nabla_\lambda L(x^*, \lambda^*) = 0 \text{ условие нахождения в бюджетном множестве}\\
& \langle y , \nabla^2_{xx} L(x^*, \lambda^*) y \rangle \geq 0, \;\;\; \forall y \in \mathbb{R}^n : \nabla h(x^*)^\top y = 0
\end{split}
$$

Сразу же заметим, что $$L(x^*, \lambda^*) = f(x^*)$$.

### General formulation

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
\text{s.t. } & h_i(x) = 0, \; i = 1,\ldots, m 
\end{split}
$$

Solution 

$$
L(x, \lambda) = f(x) + \sum\limits_{i=1}^m\lambda_i g_i(x) = f(x) + \lambda^\top g(x)
$$

Пусть $$f(x)$$ и $$h_i(x)$$ дважды дифференцируемы в точке $$x^*$$ и непрерывно дифференцируемы в некоторой окрестности $$x^*$$. Условия локального минимума для $$x \in \mathbb{R}^n, \lambda \in \mathbb{R}^m $$ запишутся как

$$
\begin{split}
& \nabla_x L(x^*, \lambda^*) = 0 \\
& \nabla_\lambda L(x^*, \lambda^*) = 0 \\
& \langle y , \nabla^2_{xx} L(x^*, \lambda^*) y \rangle \geq 0, \;\;\; \forall y \in \mathbb{R}^n : \nabla h(x^*)^\top y = 0
\end{split}
$$

В зависимости от поведения гессиана критические точки могут иметь разный характер

![](../kkt_images/critical.png)

## Optimization with inequality conditions
### Example

$$
f(x) = x_1^2 + x_2^2 \;\;\;\; g(x) = x_1^2 + x_2^2 - 1
$$

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
\text{s.t. } & g(x) \leq 0
\end{split}
$$

![](../kkt_images/KKT_p027.png)

![](../kkt_images/KKT_p028.png)

![](../kkt_images/KKT_p029.png)

![](../kkt_images/KKT_p030.png)

Таким образом, если ограничения типа неравенств неактивны в задаче БМ, то можно не парится и выписывать решение задачи БМ. Однако, так бывает не всегда:) Рассмотрим второй игрушечный пример
$$
f(x) = (x_1 - 1.1)^2 + (x_2 - 1.1)^2 \;\;\;\; g(x) = x_1^2 + x_2^2 - 1
$$

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
\text{s.t. } & g(x) \leq 0
\end{split}
$$

![](../kkt_images/KKT_p033.png)

![](../kkt_images/KKT_p034.png)

![](../kkt_images/KKT_p035.png)

![](../kkt_images/KKT_p036.png)

![](../kkt_images/KKT_p037.png)

![](../kkt_images/KKT_p038.png)

![](../kkt_images/KKT_p039.png)

![](../kkt_images/KKT_p040.png)

Итого, мы имеем проблему:

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
\text{s.t. } & g(x) \leq 0
\end{split}
$$

И два возможных случая:

1.
	$$
	\begin{split}
    & g(x^*) < 0 \\
    & \nabla f(x^*) = 0 \\
    & \nabla^2 f(x^*) > 0
    \end{split}
    $$
    
2.
	$$ \begin{split}
    & g(x^*) = 0 \\
    & - \nabla f(x^*) = \mu \nabla g(x^*), \;\; \mu > 0 \\
    & \langle y , \nabla^2_{xx} L(x^*, \mu^*) y \rangle \geq 0, \;\;\; \forall y \in \mathbb{R}^n : \nabla g(x^*)^\top y = 0
    \end{split}
    $$
   

Объединяя два возможных случая, можно записать общие условия для задачи:

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
\text{s.t. } & g(x) \leq 0 \\
&
\end{split}
$$

Определим функцию Лагранжа:

$$
L (x, \mu) = f(x) + \mu g(x)
$$

Тогда точка $$x^*$$ - локальный минимум описанной выше задачи, тогда и только тогда, когда:

$$
\begin{split}
    & (1) \; \nabla_x L (x^*, \mu^*) = 0 \\
    & (2) \; \mu^* \geq 0 \\
    & (3) \; \mu^* g(x^*) = 0 \\
    & (4) \; g(x^*) \leq 0\\
    & (5) \; \langle y , \nabla^2_{xx} L(x^*, \mu^*) y \rangle \geq 0, \;\;\; \forall y \in \mathbb{R}^n : \nabla g(x^*)^\top y = 0
\end{split}
$$

Сразу же заметим, что $$L(x^*, \mu^*) = f(x^*)$$. Условия $$\mu^* > 0 , (1), (4)$$ при этом реализуют первый сценарий, а условия $$\mu^* > 0 , (1), (3)$$ - второй.

### General formulation

$$
\begin{split}
& f(x) \to \min\limits_{x \in \mathbb{R}^n}\\
\text{s.t. } & g_i(x) \leq 0, \; i = 1,\ldots,m\\
& h_j(x) = 0, \; j = 1,\ldots, p
\end{split}
$$

Данная формулировка представляет собой общую задачу математического программирования. С этого момента и далее мы рассматриваем только $$\textbf{регулярные}$$ задачи. Это очень важное с формальной точки зрения замечание. Желающих разобраться подробнее просим обратиться к гуглу.

Solution

$$
L(x, \mu, \lambda) = f(x) + \sum\limits_{j=1}^p\lambda_j h_j(x) + \sum\limits_{i=1}^m \mu_i g_i(x)
$$

# Karush-Kuhn-Tucker conditions
{% include tabs.html bibtex = '@misc{kuhn1951nonlinear,
  title={Nonlinear programming, in (J. Neyman, ed.) Proceedings of the Second Berkeley Symposium on Mathematical Statistics and Probability},
  author={Kuhn, Harold W and Tucker, Albert W},
  year={1951},
  publisher={University of California Press, Berkeley}
}' file='/assets/files/kuhntucker.pdf' inline = 'True'%}

{% include tabs.html bibtex = '@article{karush1939minima,
  title={Minima of functions of several variables with inequalities as side constraints},
  author={Karush, William},
  journal={M. Sc. Dissertation. Dept. of Mathematics, Univ. of Chicago},
  year={1939}
}' file='/assets/files/karush.pdf' inline = 'True'%}

Пусть $$x^*$$ решение задачи математического программирования, и функции $$f, h_j, g_i$$ дифференцирумы. 
Тогда найдутся такие $$\lambda^*$$ и $$\mu^*$$, что выполнены следующие условия:

* \$$\nabla_x L(x^*, \lambda^*, \mu^*) = 0$$
* \$$\nabla_\lambda L(x^*, \lambda^*, \mu^*) = 0$$
* \$$ \mu^*_j \geq 0$$
* \$$\mu^*_j g_j(x^*) = 0$$
* \$$g_j(x^*) \leq 0$$

Эти условия являются достаточными, если задача регулярна, т. е. если:
1) данная задача есть задача выпуклой оптимизации (т. е. функции $$ f$$  и $$ g_i$$ выпуклые, $$ h_i$$ - аффинные) и выполнено условие Слейтера;
  либо
2) выполнена сильная двойственность.	

# References
* [Lecture](http://www.csc.kth.se/utbildning/kth/kurser/DD3364/Lectures/KKT.pdf) on KKT conditions (very intuitive explanation) in course "Elements of Statistical Learning" @ KTH.
* [One-line proof of KKT](https://link.springer.com/content/pdf/10.1007%2Fs11590-008-0096-3.pdf)
