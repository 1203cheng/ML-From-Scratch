# Polynomial Regression — Three Solvers from Scratch

Fits a polynomial to 2D data points using three different solvers, all built on a hand-written `Matrix` class in pure Python. No NumPy; matplotlib is used only for plotting.

## Problem

Given data points $(x, y)$, the number of polynomial bases $n$, and a regularization weight $\lambda$, find weights $w$ for

$$f(x) = w_0 + w_1 x + \dots + w_{n-1} x^{n-1}$$

## Approach

### Matrix Class

Supports addition, subtraction, scalar and matrix multiplication (`@`), transpose, and `** -1` for inversion. Inversion is done through LU decomposition followed by forward/back substitution, with row swapping when a pivot is zero.

### Three Solvers

| Solver | Objective | Update |
|--------|-----------|--------|
| **Regularized LSE** (ridge) | $\lVert Aw - y \rVert^2 + \lambda \lVert w \rVert^2$ | Closed form: $w = (A^\top A + \lambda I)^{-1} A^\top y$ |
| **Steepest Descent** (LASSO) | $\lVert Aw - y \rVert^2 + \lambda \lVert w \rVert_1$ | Subgradient $2A^\top A w - 2A^\top y + \lambda \operatorname{sign}(w)$, fixed learning rate |
| **Newton's Method** | $\lVert Aw - y \rVert^2$ | $w \leftarrow w - H^{-1} \nabla$, with Hessian $H = 2A^\top A$ |

Each solver prints the fitted polynomial and its total squared error, then plots the curve against the data.

## Usage

```bash
python regression.py   # reads testfile.txt (one "x,y" per line), then prompts for n and λ
```

## Files

- `regression.py` — Matrix class and all three solvers (~237 lines)
