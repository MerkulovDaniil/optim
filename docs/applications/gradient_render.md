---
layout: default
title: Gradient render
parent: Applications
---

# Основа

<H1>Доброе утро</H1>
<p>Рад тебя видеть</p>

<!-- <script>
	var a = 5,
		b = 7;
	var str = "блять";
	// alert(str);
	alert(a+b);
</script> -->

![](../rendezvous.svg)

We have two bodies in discrete time: the first is described by its coordinate $$x_i$$ and its speed $$v_i$$, the second has coordinate $$z_i$$ and speed $$u_i$$. Each body has its own dynamics, which we denote as linear systems with matrices $$A, B, C, D$$:

$$
\begin{align*}
x_{i+1} = Ax_i + Bu_i \\
z_{i+1} = Cz_i + Dv_i
\end{align*}
$$