# We have Double Descent at home!


<div class="responsive-video"><video autoplay loop class="video"><source src="double_descent.mp4" type="video/mp4">Your browser does not support the video tag.</video></div>

**Mom, I want double descent!**  
*“We have double descent at home.”*

The phenomenon of [double descent](https://arxiv.org/abs/1812.11118) is
a curious effect observed in machine learning models. As model
complexity increases, prediction quality on test data first improves
(*the first descent*), then, as expected, error rises due to
overfitting. But unexpectedly, beyond a certain point, the model’s
generalization ability improves again (*the second descent*).
Interestingly, a similar pattern can be observed in a relatively simple
example of polynomial regression. In the animation above, the left side
shows 50 sinusoidal points used as a training set for polynomial
regression.

A typical outcome when increasing the polynomial degree is overfitting —
almost zero error on the training data (the blue points on the left) but
high error on the test set (100 evenly spaced points along the black
line). However, as the model’s complexity continues to increase, we
observe that, out of the infinite number of possible solutions, smoother
ones are somehow selected. In this way, we clearly witness *double
descent.*

Of course, this doesn’t happen by accident. In this particular
animation, I used Chebyshev polynomials, and from the infinite set of
suitable polynomial coefficients, I chose those with the smallest norm.
In machine learning, it’s often not possible to explicitly enforce this,
so techniques like *weight decay* are used to favor solutions with
smaller norms.

[Code](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/double_descent_visualization.ipynb)
