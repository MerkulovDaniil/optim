# Why stochastic gradient descent does not converge?


Many people are used to apply SGD(stochastic gradient descent), but not
everyone knows that it is guaranteed(!) to be non convergent in the
constant stepsize (learning rate) case even for the world’s nicest
function - a strongly convex quadratic (even on average).

Why so? The point is that SGD actually solves a different problem built
on the selected data at each iteration. And this problem on the batch
may be radically different from the full problem (however, the careful
reader may note that this does not guarantee a very bad step). That is,
at each iteration we actually converge, but to the minimum of a
different problem, and each iteration we change the rules of the game
for the method, preventing it from taking more than one step.

At the end of the attached video, you can see that using selected points
from the linear regression problem, we can construct an optimal solution
to the batched problem and a gradient to it - this will be called the
stochastic gradient for the original problem. Most often the
stochasticity of SGD is analyzed using noise in the gradient and less
often we consider noise due to randomness/incompleteness of the choice
of the problem to be solved (interestingly, these are not exactly the
same thing).

This is certainly not a reason not to use the method, because
convergence to an approximate solution is still guaranteed. For convex
problems it is possible to deal with nonconvergence by \* gradually
decreasing the step (slow convergence) \* increasing the patch size
(expensive) \* using variance reduction methods (more on this later)

<div class="responsive-video"><video autoplay loop class="video"><source src="sgd_divergence.mp4" type="video/mp4">Your browser does not support the video tag.</video></div>

[Code](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/SGD_2d_visualization.ipynb)
