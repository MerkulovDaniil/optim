---
layout: default
parent: Theory
title: Conjugate set
nav_order: 4
---

# Conjugate (dual) set

Пусть $$S \in \mathbb{R}^n$$ - произвольное непустое множество. Тогда cопряженное к нему множество определяется, как:

$$
S^* = \{y \ \in \mathbb{R}^n \mid \langle y, x\rangle \ge -1 \;\; \forall x \in S\}
$$


![](../normal.png)

## Double conjugate set
Множество $$S^{**}$$ называется вторым сопряженным к множеству $$S$$, если:

$$
S^{**} = \{x \ \in \mathbb{R}^n \mid \langle y, x\rangle \ge -1 \;\; \forall x \in S^*\}
$$

## Inter-conjugate and self-conjugate sete
* Множества $$S_1$$ и $$S_2$$ называются **взаимосопряженными**, если $$S_1^* = S_2, S_2^* = S_1$$.
* Множество $$S$$ называется **самосопряженным**, если $$S^{*} = S$$

## Properties
* Сопряженное множество всегда замкнуто, выпукло и содержит нуль.
* Для произвольного множества $$S \subseteq \mathbb{R}^n$$: 

	$$
	 S^{**} = \overline{ \mathbf{conv} (S \cup  \{0\}) }
	$$

* Если $$S_1 \subset S_2$$, то $$S_2^* \subset S_1^*$$
* $$\left( \bigcup\limits_{i=1}^m S_i \right)^* = \bigcap\limits_{i=1}^m S_i^*$$
* Если $$S$$ - замкнуто, выпукло, включает $$0$$, то $$S^{**} = S$$
* $$S^* = \left(\overline{S}\right)^*$$

## Examples

### 1
Доказать, что $$S^* = \left(\overline{S}\right)^*$$

Решение:

* $$S \subset \overline{S} \rightarrow \left(\overline{S}\right)^* \subset S^*$$
* Пусть $$p \in S^*$$ и $$x_0 \in \overline{S}, x_0 = \underset{k \to \infty}{\operatorname{lim}} x_k$$. Тогда в силу непрерывности функции $$f(x) = p^Tx$$, имеем: $$p^T x_k \ge -1 \to p^Tx_0 \ge -1$$. Значит, $$p \in  \left(\overline{S}\right)^*$$, отсюда $$S^* \subset \left(\overline{S}\right)^*$$

### 2
Доказать, что $$\left( \mathbf{conv}(S) \right)^* = S^*$$

Решение:

* $$S \subset \mathbf{conv}(S) \to \left( \mathbf{conv}(S) \right)^* \subset S^*$$
* Пусть $$p \in S^*$$, $$x_0 \in \mathbf{conv}(S)$$, т.е. $$x_0 = \sum\limits_{i=1}^k\theta_i x_i \mid x_i \in S, \sum\limits_{i=1}^k\theta_i = 1, \theta_i \ge 0$$.

    Значит, $$p^T x_0 = \sum\limits_{i=1}^k\theta_i p^Tx_i \ge \sum\limits_{i=1}^k\theta_i (-1) = 1 * (-1) = -1$$. Значит, $$p \in  \left( \mathbf{conv}(S) \right)^*$$, отсюда $$S^* \subset \left( \mathbf{conv}(S) \right)^*$$

### 3
Доказать, что если $$B(0,r)$$ - шар радиуса $$r$$ по некоторой норме с центром в нуле, то $$\left( B(0,r) \right)^* = B(0,1/r)$$

Решение:

* Пусть $$B(0,r) = X, B(0,1/r) = Y$$. Возьмем вектор нормали $$p \in X^*$$, тогда для любого $$x \in X: p^Tx \ge -1$$
* Из всех точек шара $$X$$ возьмем такую $$x \in X$$, что скалярное произведение её на $$p$$: $$p^Tx$$ было бы минимально, тогда это точка $$x = -\frac{p}{\|p\|}r$$

	$$
	p^T x = p^T \left(-\frac{p}{\|p\|}r \right) = -\|p\|r \ge -1
	$$

	$$
	\|p\| \le \frac{1}{r} \in Y
	$$

	Значит, $$X^* \subset Y$$
* Теперь пусть $$p \in Y$$. Нам надо показать, что $$p \in X^*$$, т.е. $$\langle p, x\rangle \geq -1$$. Достаточно применить неравенство Коши - Буняковского:

	$$
	\|\langle p, x\rangle\| \leq \|p\| \|x\| \leq \dfrac{1}{r} \cdot r = 1
	$$

    Последнее исходит из того, что $$p \in B(0,1/r)$$, а $$x \in B(0,r)$$
    
    Значит, $$Y \subset X^*$$

# Dual cones
Сопряженным конусом к конусу $$K$$ называется такое множество $$K^*$$, что: 

$$
K^* = \left\{ y \mid \langle x, y\rangle \ge 0 \;\;\;\; \forall x \in K\right\}
$$

Чтобы показать, что это определение непосредственно следует из теории выше вспомним, что такое сопряженное множество и что такое конус $$\forall \lambda > 0$$

$$
\{y \ \in \mathbb{R}^n \mid \langle y, x\rangle \ge -1 \;\; \forall x \in S\} \to \{\lambda y \ \in \mathbb{R}^n \mid \langle y, x\rangle \ge -\dfrac{1}{\lambda} \;\; \forall x \in S\}
$$

![](../dual_cone.gif)

## Dual cones properties
* Если $$K$$ - замкнутый выпуклый конус. Тогда $$K^{**} = K$$
* Для произвольного множества $$S \subseteq \mathbb{R}^n$$ и конуса $$K \subseteq \mathbb{R}^n$$: 

	$$
	\left( S + K \right)^* = S^* \cap K^*
	$$

* Пусть $$K_1, \ldots, K_m$$ - конусы в $$\mathbb{R}^n$$, тогда:

	$$
	\left( \sum\limits_{i=1}^m K_i \right)^* = \bigcap\limits_{i=1}^m K_i^*
	$$

* Пусть $$K_1, \ldots, K_m$$ - конусы в $$\mathbb{R}^n$$. Пусть так же, их пересечение имеет внутреннюю точку, тогда:

	$$
	\left( \bigcap\limits_{i=1}^m K_i \right)^* = \sum\limits_{i=1}^m K_i^*
	$$

## Examples

Найти сопряженнй конус для монотонного неотрицательного конуса: 

$$
 K = \left\{ x \in \mathbb{R}^n \mid x_1 \ge x_2 \ge \ldots \ge x_n \ge 0\right\}
$$

Решение:

Заметим, что: 

$$
\sum\limits_{i=1}^nx_iy_i = y_1 (x_1-x_2) + (y_1 + y_2)(x_2 - x_3) + \ldots + (y_1 + y_2 + \ldots + y_{n-1})(x_{n-1} - x_n) + (y_1 + \ldots + y_n)x_n
$$

Так как в представленной сумме в каждом слагаемом второй множитель положительный, то:

$$
y_1 \ge 0, \;\; y_1 + y_2 \ge 0, \;\;\ldots, \;\;y_1 + \ldots + y_n \ge 0
$$

Значит, $$K^* = \left\{ y \mid \sum\limits_{i=1}^k y_i \ge 0, k = \overline{1,n}\right\}$$

## Polyhedra
Множество решений системы линейных неравенств и равенств представляет собой многогранник:

$$
Ax \preceq b, \;\;\; Cx = d
$$

Здесь $$A \in \mathbb{R}^{m\times n}, C \in \mathbb{R}^{p \times n} $$, а неравенство - поэлементное.

![](../polyhedra.svg)

#### Теорема:
Пусть $$x_1, \ldots, x_m \in \mathbb{R}^n$$. Сопряженным к многогранному множеству:

$$
 S = \mathbf{conv}(x_1, \ldots, x_k) + \mathbf{cone}(x_{k+1}, \ldots, x_m) 
$$

является полиэдр (многогранник):

$$
 S^* = \left\{ p \in \mathbb{R}^n \mid \langle p, x_i\rangle \ge -1, i = \overline{1,k} ; \langle p, x_i\rangle \ge 0, i = \overline{k+1,m} \right\}
$$

#### Доказательство:

* Пусть $$S = X, S^* = Y$$. Возьмем некоторый $$p \in X^*$$, тогда $$\langle p, x_i\rangle \ge -1,  i = \overline{1,k}$$. В то же время для любых $$\theta > 0, i = \overline{k+1,m}$$: 
  
	$$
	\langle p, x_i\rangle \ge -1 \to \langle p, \theta x_i\rangle \ge -1
	$$

	$$
	\langle p, x_i\rangle \ge -\frac{1}{\theta} \to \langle p, x_i\rangle \ge 0 
	$$

	Значит, $$p \in Y \to X^* \subset Y$$

* Пусть, напротив, $$p \in Y$$. Для любой точки $$x \in X$$:

	$$
	 x = \sum\limits_{i=1}^m\theta_i x_i \;\;\;\;\;\; \sum\limits_{i=1}^k\theta_i = 1, \theta_i \ge 0
	$$
  
	Значит:

	$$
	\langle p, x\rangle = \sum\limits_{i=1}^m\theta_i \langle p, x_i\rangle  = \sum\limits_{i=1}^k\theta_i \langle p, x_i\rangle + \sum\limits_{i=k+1}^m\theta_i \langle p, x_i\rangle \ge \sum\limits_{i=1}^k\theta_i (-1) + \sum\limits_{i=1}^k\theta_i \cdot 0 = -1
	$$

	Значит, $$p \in X^* \to Y \subset X^*$$

### 5

Найти и изобразить на плоскости множество, сопряженное к многогранному конусу:

$$
 S = \mathbf{cone} \left\{ (-3,1), (2,3), (4,5)\right\} 
$$

Решение:

Используя теорему выше: 

$$
 S^* = \left\{ -3p_1 + p_2 \ge 0, 2p_1 + 3p_2 \ge 0, 4p_1+5p_2 \ge 0 \right\} 
$$

### Лемма (теорема) Фаркаша (Фаркаша - Минковского)
Пусть $$A \in \mathbb{R}^{m\times n}, b \in \mathbb{R}^m$$. Тогда имеет решение одна и только одна из следующих двух систем: 

$$
1) \; Ax = b, x \ge 0\;\;\;\;\;\;
$$


$$
2) \; pA \ge 0, \langle p,b\rangle < 0
$$


$$Ax = b$$ при $$x \geq 0$$ означает, что $$b$$ лежит в конусе, натянутым на столбцы матрицы $$A$$

$$pA \geq 0, \; \langle p, b \rangle < 0$$ означает, что существует разделяющая гиперплоскость между вектором $$b$$ и конусом из столбцов матрицы $$A$$.

#### Следствие: 
Пусть $$A \in \mathbb{R}^{m\times n}, b \in \mathbb{R}^m$$. Тогда имеет решение одна и только одна из следующих двух систем: 

$$
1) \; Ax \le b \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;
$$

$$
2) \; pA = 0, \langle p,b\rangle < 0, p \ge 0
$$

Если в задаче линейного программирования на минимум допустимое множество непусто и целевая функция ограничена на нём снизу, то задача имеет решение.