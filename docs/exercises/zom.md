---
layout: default
title: Zero order methods
parent: Exercises
nav_order: 15
---

# SciPy library
1. Implement Rastrigin function $f: \mathbb{R}^d \to \mathbb{R}$ for d = 10. [link](https://www.sfu.ca/~ssurjano/rastr.html)

	$$
	f(\mathbf{x})=10 d+\sum_{i=1}^{d}\left[x_{i}^{2}-10 \cos \left(2 \pi x_{i}\right)\right]
	$$

	* Consider global optimization from [here](https://docs.scipy.org/doc/scipy/reference/optimize.html#global-optimization).
	* Plot 4 graphs for different $d$ from {10, 100, 1000, 10000}. On each graph you are to plot $f$ from $N_{fev}$ for 5 methods: `basinhopping`, `brute`, `differential_evolution`, `shgo`, `dual_annealing` from scipy, where $N_{fev}$ - the number of function evaluations. This information is usually available from `specific_optimizer.nfev`. If you will need bounds for the optimizer, use $x_i \in [-5, 5]$.

1. Machine learning models often have hyperparameters. To choose optimal one between them one can use GridSearch or RandomSearch. But these algorithms computationally uneffective and don't use any sort of information about type of optimized function. To overcome this problem one can use [bayesian optimization](https://distill.pub/2020/bayesian-optimization/). Using this method we optimize our model by sequentially chosing points based on prior information about function. ![Image](https://www.resibots.eu/limbo/_images/bo_concept.png). 

	In this task you will use [optuna](https://optuna.org/) package for hyperparameter optimization RandomForestClassifier. Your task is to find best Random Forest model varying at least 3 hyperparameters on iris dataset. Examples can be find [here](https://optuna.org/#code_examples) or [here](www.kaggle.com/dixhom/bayesian-optimization-with-optuna-stacking/)

	```bash
	!pip install optuna
	```

	```python
	import sklearn.datasets
	import sklearn.ensemble
	import sklearn.model_selection
	import sklearn.svm

	import optuna

	iris = sklearn.datasets.load_iris()
	x, y = iris.data, iris.target
	```
1. Try to perform hyperparameter optimization in context of any metric for imbalanced classification problem with optuna and keras. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/optuna_keras.ipynb)
