---
layout: default
title: First order methods
parent: Exercises
nav_order: 16
---

1. A function is said to belong to the class $f \in C^{k,p}_L (Q)$ if it $k$ times is continuously differentiable on $Q$ and the $p$th derivative has a Lipschitz constant $L$. 

    $$
    \|\nabla^p f(x) - \nabla^p f(y)\| \leq L \|x-y\|, \qquad \forall x,y \in Q
    $$

    The most commonly used $C_L^{1,1}, C_L^{2,2}$ on $\mathbb{R}^n$. 
    Note that:
    * $p \leq k$
    * If $q \geq k$, then $C_L^{q,p} \subseteq C_L^{k,p}$. The higher the order of the derivative, the stronger the constraint (fewer functions belong to the class)

    Prove that the function belongs to the class $C_L^{2,1} \subseteq C_L^{1,1}$ if and only if $\forall x \in \mathbb{R}^n$:

    $$
    \||\nabla^2 f(x)\| \leq L
    $$

    Prove also that the last condition can be rewritten, without generality restriction, as follows:

    $$
    -L I_n \preceq \nabla^2 f(x) \preceq L I_n
    $$

    Note: by default the Euclidean norm is used for vectors and the spectral norm is used for matrices.

1. Покажите, что с помощью следующих стратегий подбора шага в градиентному спуске:
    * Постоянный шаг $h_k = \dfrac{1}{L}$
    * Убывающая последовательность $h_k = \dfrac{\alpha_k}{L}, \quad \alpha_k \to 0$

    можно получить оценку убывания функции на итерации вида:

    $$
    f(x_k) - f(x_{k+1}) \geq \dfrac{\omega}{L}\|\nabla f(x_k)\|^2
    $$

    $\omega > 0$ - некоторая константа, $L$ - константа Липщица градиента функции 

1. Рассмотрим функцию двух переменных:

    $$
    f(x_1, x_2) = x_1^2 + k x_2^2,
    $$

    где $k$ - некоторый параметр. Постройте график количества итераций, необходимых для сходимости алгоритма наискорейшего спуска (до выполнения условия $\|\nabla f(x_k)\| \leq \varepsilon = 10^{-7}$) в зависимости от значения $k$. Рассмотрите интервал $k \in [10^{-3}; 10^3]$ (будет удобно использовать функцию `ks = np.logspace(-3,3)`) и строить график по оси абсцисс в логарифмическом масштабе `plt.semilogx()` или `plt.loglog()` для двойного лог. масштаба.

    Сделайте те же графики для функции:

    $$
    f(x) = \ln(1 + e^{x^\top A x}) + \mathbf{1}^\top x
    $$

    Объясните полученную зависимость.

    Для наглядности можете пользоваться кодом отрисовки картинок:

    ```python
    def f_6(x, *f_params):
        if len(f_params) == 0:
            k = 2
        else:
            k = float(f_params[0])
        x_1, x_2 = x
        return x_1**2 + k*x_2**2

    def df_6(x, *f_params):
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

    plot_3d_function(x1, x2, f_6, title, k, minima=[0,0])

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
    iterations = steepest_descent(x_0, f_6, df_6, k, df_eps = 1e-9)
    title = f'$f(x_1, x_2) = x_1^2 + k x_2^2, k = {k}$'

    plot_3d_function(x1, x2, f_6, title, k, minima=[0,0], iterations = iterations)
    ```

1. Solve the Hobbit Village problem. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Hobbit_village.ipynb)

1. Solve the problem of constrained optimization using projected gradient descent [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Projected_gradient_descent_affine.ipynb).
