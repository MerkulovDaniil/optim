---
layout: default
parent: Convex sets
grand_parent: Theory
title: Projection
nav_order: 5
---

# Projection
## Distance between point and set
Расстоянием $$d$$ от точки $$\mathbf{y} \in \mathbb{R}^n$$ до замкнутого множества $$S \subset \mathbb{R}^n$$ является:

$$
d(\mathbf{y}, S, \| \cdot \|) = \inf\{\|x - y\| \mid x \in S \}
$$


## Projection of a point on set
Проекцией точки $$\mathbf{y} \in \mathbb{R}^n$$ на множество $$S \subseteq \mathbb{R}^n$$ называется точка $$\pi_S(\mathbf{y}) \in S$$: 

$$
\| \pi_S(\mathbf{y}) - \mathbf{y}\| \le \|\mathbf{x} - \mathbf{y}\|, \forall \mathbf{x} \in S
$$

* Если множество - открыто, и точка в нем не лежит, то её проекции на это множество не существует
* Если точка лежит в множестве, то её проекция - это сама точка
* 	$$
	\pi_S(\mathbf{y}) = \underset{\mathbf{x}}{\operatorname{argmin}} \|\mathbf{x}-\mathbf{y}\|
	$$

* Пусть $$S \subseteq \mathbb{R}^n$$ - выпуклое замкнутое множество. Пусть так же имеются точки $$\mathbf{y} \in \mathbb{R}^n$$ и $$\mathbf{\pi} \in S$$. Тогда если для всех $$\mathbf{x} \in S$$ справедливо неравенство: 
	
	$$
	\langle \pi  -\mathbf{y}, \mathbf{x} - \pi\rangle \ge 0, 
	$$

	то $$\pi$$ является проекцией точки $$\mathbf{y}$$ на $$S$$, т.е. $$\pi_S (\mathbf{y}) = \pi$$ 
* Пусть $$S \subseteq \mathbb{R}^n$$ - афинное множество. Пусть так же имеются точки $$\mathbf{y} \in \mathbb{R}^n$$ и $$\mathbf{\pi} \in S$$. Тогда $$\pi$$ является проекцией точки $$\mathbf{y}$$ на $$S$$, т.е. $$\pi_S (\mathbf{y}) = \pi$$ тогда и только тогда, когда для всех $$\mathbf{x} \in S$$ справедливо равенство: 

$$
\langle \pi  -\mathbf{y}, \mathbf{x} - \pi\rangle = 0 
$$
 
#### Example 1
Найти $$\pi_S (y) = \pi$$, если $$S = \{x \in \mathbb{R}^n \mid \|x - x_c\| \le R \}$$, $$y \notin S$$ 

Решение:
![](../proj_cir.gif)

* Из рисунка строим гипотезу: $$\pi = x_0 + R \cdot \frac{y - x_0}{\|y - x_0\|}$$ 

* Проверяем неравенство для выпуклого замкнутого множества: $$(\pi - y)^T(x - \pi) \ge 0$$ 

	$$
	\left( x_0 - y + R \frac{y - x_0}{\|y - x_0\|} \right)^T\left( x - x_0 - R \frac{y - x_0}{\|y - x_0\|} \right) =
	$$

	$$
	\left( \frac{(y - x_0)(R - \|y - x_0\|)}{\|y - x_0\|} \right)^T\left( \frac{(x-x_0)\|y-x_0\|-R(y - x_0)}{\|y - x_0\|} \right) =
	$$

	$$
	\frac{R - \|y - x_0\|}{\|y - x_0\|^2} \left(y - x_0 \right)^T\left( \left(x-x_0\right)\|y-x_0\|-R\left(y - x_0\right) \right) = 
	$$

	$$
	\frac{R - \|y - x_0\|}{\|y - x_0\|} \left( \left(y - x_0 \right)^T\left( x-x_0\right)-R\|y - x_0\| \right) =
	$$

	$$
	\left(R - \|y - x_0\| \right) \left( \frac{(y - x_0 )^T( x-x_0)}{\|y - x_0\|}-R \right)
	$$

* Первый сомножитель отрицателен по выбору точки $$y$$. Второй сомножитель так же отрицателен, если применить к его записи теорему Коши - Буняковского: 

	$$
	(y - x_0 )^T( x-x_0) \le \|y - x_0\|\|x-x_0\|
	$$

	$$
	\frac{(y - x_0 )^T( x-x_0)}{\|y - x_0\|} - R \le \frac{\|y - x_0\|\|x-x_0\|}{\|y - x_0\|} - R = \|x - x_0\| - R \le 0
	$$

#### Example 2
Найти $$\pi_S (y) = \pi$$, если $$S = \{x \in \mathbb{R}^n \mid c^T x = b \}$$, $$y \notin S$$ 

Решение:

![](../proj_half.gif)

* Из рисунка строим гипотезу: $$\pi = y + \alpha c$$. Коэффициент $$\alpha$$ подбирается так, чтобы $$\pi \in S$$: $$c^T \pi = b$$, т.е.: 

	$$
	c^T (y + \alpha c) = b
	$$

	$$
	c^Ty + \alpha c^T c = b
	$$

	$$
	c^Ty = b - \alpha c^T c
	$$

* Проверяем неравенство для выпуклого замкнутого множества: $$(\pi - y)^T(x - \pi) \ge 0$$ 

	$$
	(y + \alpha c - y)^T(x - y - \alpha c) = 
	$$

	$$
	 \alpha c^T(x - y - \alpha c) = 
	$$

	$$
	 \alpha (c^Tx) - \alpha (c^T y) - \alpha^2 c^Tc) = 
	$$

	$$
	 \alpha b - \alpha (b - \alpha c^T c) - \alpha^2 c^Tc = 
	$$

	$$
	 \alpha b - \alpha b + \alpha^2 c^T c - \alpha^2 c^Tc = 0 \ge 0
	$$

#### Example 3
Найти $$\pi_S (y) = \pi$$, если $$S = \{x \in \mathbb{R}^n \mid Ax = b, A \in \mathbb{R}^{m \times n},  b \in \mathbb{R}^{m} \}$$, $$y \notin S$$ 

Решение:

![](../proj_poly.gif)

* Из рисунка строим гипотезу: $$\pi = y + \sum\limits_{i=1}^m\alpha_i A_i = y + A^T \alpha$$. Коэффициент $$\alpha$$ подбирается так, чтобы $$\pi \in S$$: $$A \pi = b$$, т.е.: 

	$$
	c^T (y + A^T \alpha) = b
	$$

	$$
	A(y + A^T\alpha) = b
	$$

	$$
	Ay = b - A A^T\alpha
	$$

* Проверяем неравенство для выпуклого замкнутого множества: $$(\pi - y)^T(x - \pi) \ge 0$$ 

	$$
	(y + A^T\alpha  - y)^T(x - y - A^T\alpha) = 
	$$

	$$
	 \alpha^T A(x - y - A^T\alpha) = 
	$$

	$$
	 \alpha^T (Ax) - \alpha^T (A y) - \alpha^T AA^T \alpha) = 
	$$

	$$
	 \alpha^T b - \alpha^T (b - A A^T\alpha) - \alpha^T AA^T \alpha = 
	$$

	$$
	 \alpha^T b - \alpha^T b + \alpha^T AA^T \alpha - \alpha^T AA^T \alpha = 0 \ge 0
	$$ 
