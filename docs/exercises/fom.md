---
layout: default
title: First order methods
parent: Exercises
nav_order: 16
---

1. A function is said to belong to the class $$f \in C^{k,p}_L (Q)$$ if it $$k$$ times is continuously differentiable on $$Q$$ and the $$p$$th derivative has a Lipschitz constant $$L$$. 

	$$
	\||nabla^p f(x) - \nabla^p f(y)\| \leq L \|x-y\|, \qquad \forall x,y \in Q
	$$

	The most commonly used $$C_L^{1,1}, C_L^{2,2}$$ on $$\mathbb{R}^n$$. 
	Note that:
	* $$p \leq k$$
	* If $$q \geq k$$, then $$C_L^{q,p} \subseteq C_L^{k,p}$$. The higher the order of the derivative, the stronger the constraint (fewer functions belong to the class)

	Prove that the function belongs to the class $$C_L^{2,1} \subseteq C_L^{1,1}$$ if and only if $$\forall x \in \mathbb{R}^n$$:

	$$
	\||\nabla^2 f(x)\| \leq L
	$$

	Prove also that the last condition can be rewritten, without generality restriction, as follows:

	$$
	-L I_n \preceq \nabla^2 f(x) \preceq L I_n
	$$

	Note: by default the Euclidean norm is used for vectors and the spectral norm is used for matrices.

1. Consider a function of two variables:

	$$
	f(x_1, x_2) = x_1^2 + k x_2^2,
	$$

	where $k$ is some parameter

	```python
	def f(x, *f_params):
		if len(f_params) == 0:
			k = 2
		else:
			k = float(f_params[0])
		x_1, x_2 = x
		return x_1**2 + k*x_2**2

	def df(x, *f_params):
		if len(f_params) == 0:
			k = 2
		else:
			k = float(f_params[0])
		return np.array([2*x[0], 2*k*x[1]])


	%matplotlib inline
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	from matplotlib.ticker import LinearLocator, FormatStrFormatter
	import numpy as np

	def plot_3d_function(x1, x2, f, title, *f_params, minima = None, iterations = None):
		'''
		'''
		low_lim_1 = x1.min()
		low_lim_2 = x2.min()
		up_lim_1  = x1.max()
		up_lim_2  = x2.max()

		X1,X2 = np.meshgrid(x1, x2) # grid of point
		Z = f((X1, X2), *f_params) # evaluation of the function on the grid
		
		# set up a figure twice as wide as it is tall
		fig = plt.figure(figsize=(16,7))
		fig.suptitle(title)

		#===============
		#  First subplot
		#===============
		# set up the axes for the first plot
		ax = fig.add_subplot(1, 2, 1, projection='3d')

		# plot a 3D surface like in the example mplot3d/surface3d_demo
		surf = ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, 
							cmap=cm.RdBu,linewidth=0, antialiased=False)

		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
		if minima is not None:
			minima_ = np.array(minima).reshape(-1, 1)
			ax.plot(*minima_, f(minima_), 'r*', markersize=10)
		
		

		#===============
		# Second subplot
		#===============
		# set up the axes for the second plot
		ax = fig.add_subplot(1, 2, 2)

		# plot a 3D wireframe like in the example mplot3d/wire3d_demo
		im = ax.imshow(Z,cmap=plt.cm.RdBu,  extent=[low_lim_1, up_lim_1, low_lim_2, up_lim_2])
		cset = ax.contour(x1, x2,Z,linewidths=2,cmap=plt.cm.Set2)
		ax.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
		fig.colorbar(im)
		ax.set_xlabel(f'$x_1$')
		ax.set_ylabel(f'$x_2$')
		
		if minima is not None:
			minima_ = np.array(minima).reshape(-1, 1)
			ax.plot(*minima_, 'r*', markersize=10)
		
		if iterations is not None:
			for point in iterations:
				ax.plot(*point, 'go', markersize=3)
			iterations = np.array(iterations).T
			ax.quiver(iterations[0,:-1], iterations[1,:-1], iterations[0,1:]-iterations[0,:-1], iterations[1,1:]-iterations[1,:-1], scale_units='xy', angles='xy', scale=1, color='blue')

		plt.show()

	up_lim  = 4
	low_lim = -up_lim
	x1 = np.arange(low_lim, up_lim, 0.1)
	x2 = np.arange(low_lim, up_lim, 0.1)
	k=0.5
	title = f'$f(x_1, x_2) = x_1^2 + k x_2^2, k = {k}$'

	plot_3d_function(x1, x2, f, title, k, minima=[0,0])
	```

	For example, steepest descent algorithm will be plotted with the following code:

	```python
	from scipy.optimize import minimize_scalar

	def steepest_descent(x_0, f, df, *f_params, df_eps = 1e-2, max_iter = 1000):
		iterations = []
		x = np.array(x_0)
		iterations.append(x)
		while np.linalg.norm(df(x, *f_params)) > df_eps and len(iterations) <= max_iter:
			res = minimize_scalar(lambda alpha: f(x - alpha * df(x, *f_params), *f_params))
			alpha_opt = res.x
			x = x - alpha_opt * df(x, *f_params)
			iterations.append(x)
		print(f'Finished with {len(iterations)} iterations')
		return iterations

	x_0 = [10,1]
	k = 30
	iterations = steepest_descent(x_0, f, df, k, df_eps = 1e-9)
	title = f'$f(x_1, x_2) = x_1^2 + k x_2^2, k = {k}$'

	plot_3d_function(x1, x2, f, title, k, minima=[0,0], iterations = iterations)
	```

	Plot the number of iterations required for the convergence of the steepest descent algorithm (up to the condition $\|nabla f(x_k)\| \leq \varepsilon = 10^{-7}$) depending on the value of $k$. Consider the interval $k \in [10^{-3}; 10^3]$ (it will be convenient to use the function `ks = np.logspace(-3,3)`) and plot on the X axis in logarithmic scale `plt.semilogx()` or `plt.loglog()` for double log scale.

	Make the same graphs for the suitable constant stepsize. Explain the results.