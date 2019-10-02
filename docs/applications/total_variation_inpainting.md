---
layout: default
title: Total variation in-painting
parent: Applications
---

# Problem
![](../tv_start.png)

## Grayscale image
A grayscale image is represented as an $$m \times n$$ matrix of intensities $$U^{orig}$$ (typically between the values $$0$$ and $$255$$). We are given all the values of corrupted picture, but some of them should be preserved as is through the recovering procedure: $$U^{corr}_{ij} \; \forall (i,j)\in K$$, where $$K\subset\{1,\ldots,m\}×\{1,\ldots,n\}$$ is the set of indices corresponding to known pixel values. Our job is to in-paint the image by guessing the missing pixel values, i.e., those with indices not in $$K$$. The reconstructed image will be represented by $$U \in \mathbb{R}^{m \times n}$$, where $$U$$ matches the known pixels, i.e., $$U_{ij}=U^{corr}_{ij}$$ for $$(i,j)\in K$$.

The reconstruction $$U$$ is found by minimizing the total variation of $$U$$, subject to matching the known pixel values. We will use the $$l_{2}$$ total variation, defined as

$$
\begin{split}\mathop{\bf tv}(U) =
\sum_{i=1}^{m-1} \sum_{j=1}^{n-1}
\left\| \left[ \begin{array}{c}
 U_{i+1,j}-U_{ij}\\ U_{i,j+1}-U_{ij} \end{array} \right] \right\|_2.\end{split}
$$

So, the final optimization problem will be written as follows:

$$
\begin{split}
& \mathop{\bf tv}(U) \to \min\limits_{U \in \mathbb{R}^{m \times n}} \\
\text{s.t. } & U_{ij} = U^{corr}_{ij}, \; (i,j)\in K
\end{split}
$$

The crucial thing about this problem is defining set of known pixels $$K$$. There are some heuristics: for example, we could state, that each pixel with color similar (or exactly equal) to the color of text is unknown. The results for such approach are presented below:

![](../tv_finish.png)

## Color image

For the color case we consider in-painting problem in a slightly different setting: destroying some random part of all pixels. In this case the image itself is 3d tensor (we convert all others color chemes to the RGB). As it was in the grayscale case, we construct the mask $$K$$ of known pixels for all color channels uniformly, based on the principle of similarity of particular 3d pixel to the vector $$[0, 0, 0]$$ (black pixel). The results are quite promising - note, that we have no information about the original picture, but assumption, that corrupted pixels are black. For the color picture we just sum all tv's on the each channel:

$$
\begin{split}\mathop{\bf tv}(U) =
\sum_{k = 1}^{3}\sum_{i=1}^{m-1} \sum_{j=1}^{n-1}
\left\| \left[ \begin{array}{c}
 U^k_{i+1,j}-U^k_{ij}\\ U^k_{i,j+1}-U^k_{ij} \end{array} \right] \right\|_2.\end{split}
$$

Then, we need to write down optimization problem to be solved:

$$
\begin{split}
& \mathop{\bf tv}(U) \to \min\limits_{U \in \mathbb{R}^{m \times n \times 3}} \\
\text{s.t. } & U^k_{ij} = U^{corr, k}_{ij}, \; (i,j)\in K, \; k = 1,2,3
\end{split}
$$

Results are presented below (these computations are really take time): 
![](../tv_start1.png)
![](../tv_finish1.png)

It is not that easy, right?
![](../tv_start2.png)
![](../tv_finish2.png)

Only 5% of all pixels are left:

![](../tv_start3.png)
![](../tv_finish3.png)

What about 1% of all pixels?

![](../tv_start4.png)
![](../tv_finish4.png)

# Code
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Total%20variation%20inpainting.ipynb)
# References
* [CVXPY documentation](https://www.cvxpy.org/examples/applications/tv_inpainting.html)
* [Interactive demo](https://remi.flamary.com/demos/proxtv.html)