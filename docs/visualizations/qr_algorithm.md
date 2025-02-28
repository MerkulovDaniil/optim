---
title: "QR Algorithm: Eigenvalue Computation"
---

The QR algorithm is one of the most elegant and powerful methods in numerical linear algebra for computing eigenvalues of matrices. The algorithm's beauty lies in its simplicity and theoretical depth: through iterative QR decompositions, it converges to a triangular (or diagonal for symmetric matrices) form, revealing the eigenvalues along the diagonal.

## Algorithm Overview

The basic QR iteration proceeds as follows:

1. Start with a matrix A₀ = A
2. For k = 0, 1, 2, ...:
   - Compute QR decomposition: Aₖ = QₖRₖ
   - Form next iterate: Aₖ₊₁ = RₖQₖ

For symmetric matrices, this process converges to a diagonal matrix containing the eigenvalues. For general matrices, it converges to an upper triangular (Schur) form, where the diagonal elements are still the eigenvalues. The animation below was inspired by the Gabriel Peyré [post](https://x.com/gabrielpeyre/status/1881582504219734216?t=romY906gYOjWRDboK42_Cw&s=09).

:::{.video}
qr_algorithm.mp4
:::

The visualization above demonstrates how the QR algorithm gradually transforms a symmetric matrix into diagonal form, with the off-diagonal elements converging to zero. The convergence rate depends on the separation of eigenvalues - better separated eigenvalues typically lead to faster convergence.

## Implementation

[Code](qr_algorithm.py)

## Key Properties

1. **Preservation of Eigenvalues**: Each iteration Aₖ₊₁ = RₖQₖ is similar to Aₖ, thus preserving eigenvalues
2. **Convergence**: For symmetric matrices, the algorithm converges to a diagonal matrix
3. **Numerical Stability**: The use of orthogonal transformations ensures good numerical properties

The QR algorithm forms the basis of modern eigenvalue computations and is implemented in most numerical linear algebra libraries. While the basic version shown here is elegant, practical implementations use various enhancements like shifts and deflation to improve efficiency. 