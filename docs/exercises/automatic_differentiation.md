---
layout: default
title: Automatic differentiation
parent: Exercises
nav_order: 15
---

# Automatic differentiation

1. Calculate the gradient of a Taylor series of a $\cos (x)$ using `autograd` library:
	```python
	import autograd.numpy as np # Thinly-wrapped version of Numpy 
	from autograd import grad 

	def taylor_cosine(x): # Taylor approximation to cosine function 
	  # Your np code here
	  return ans 
	```

1. In the following code for the gradient descent for linear regression change the manual gradient computation to the PyTorch/jax autograd way. Compare those two approaches in time.

	In order to do this, set the tolerance rate for the function value $$\varepsilon = 10^{-9}$$. Compare the total time required to achieve the specified value of the function for analytical and automatic differentiation. Perform measurements for different values of $$n$$ from `np.logspace(1,4)`. 

	For each $$n$$ value carry out at least 3 runs.
	
	```python
	import numpy as np 

	# Compute every step manually

	# Linear regression
	# f = w * x 

	# here : f = 2 * x
	X = np.array([1, 2, 3, 4], dtype=np.float32)
	Y = np.array([2, 4, 6, 8], dtype=np.float32)

	w = 0.0

	# model output
	def forward(x):
	    return w * x

	# loss = MSE
	def loss(y, y_pred):
	    return ((y_pred - y)**2).mean()

	# J = MSE = 1/N * (w*x - y)**2
	# dJ/dw = 1/N * 2x(w*x - y)
	def gradient(x, y, y_pred):
	    return np.dot(2*x, y_pred - y).mean()

	print(f'Prediction before training: f(5) = {forward(5):.3f}')

	# Training
	learning_rate = 0.01
	n_iters = 20

	for epoch in range(n_iters):
	    # predict = forward pass
	    y_pred = forward(X)

	    # loss
	    l = loss(Y, y_pred)
	    
	    # calculate gradients
	    dw = gradient(X, Y, y_pred)

	    # update weights
	    w -= learning_rate * dw

	    if epoch % 2 == 0:
	        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
	     
	print(f'Prediction after training: f(5) = {forward(5):.3f}')
	``` 

1. Calculate the 4th derivative of hyperbolic tangent function using `Jax` autograd.
1. Compare analytic and autograd (with any framework) approach for the hessian of:		
	
	$$
	f(x) = \dfrac{1}{2}x^TAx + b^Tx + c
	$$

1. Compare analytic and autograd (with any framework) approach for the gradient of:		
	
	$$
	f(X) = tr(AXB)
	$$

1. Compare analytic and autograd (with any framework) approach for the gradient and hessian of:		
	
	$$
	f(x) = \dfrac{1}{2} \|Ax - b\|^2_2
	$$

1. Compare analytic and autograd (with any framework) approach for the gradient and hessian of:		
	
	$$
	f(x) = \ln \left( 1 + \exp\langle a,x\rangle\right) 
	$$

# Materials
* [HIPS autograd](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md)
* [PyTorch autograd](https://www.python-engineer.com/courses/pytorchbeginner/05-gradient-descent/)
* [Jax Autodiff cookbook](https://colab.research.google.com/github/google/jax/blob/master/docs/notebooks/autodiff_cookbook.ipynb#scrollTo=RsQSyT5p7OJW)
