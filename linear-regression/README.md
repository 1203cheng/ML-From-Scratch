# Polynomial Regression — Three Solvers from Scratch

Fits a polynomial to 2D data points using three different solvers, all built on a hand-written `Matrix` class in pure Python. No NumPy; matplotlib is used only for plotting.

## Problem

Given data points `(x, y)`, the number of polynomial bases `n`, and a regularization weight `λ`, find weights `w` for `f(x) = w₀ + w₁x + … + wₙ₋₁xⁿ⁻¹`.

## Approach

### Matrix Class

Supports addition, subtraction, scalar and matrix multiplication (`@`), transpose, and `** -1` for inversion. Inversion is done through LU decomposition followed by forward/back substitution, with row swapping when a pivot is zero.

### Three Solvers

| Solver | Objective | Update |
|--------|-----------|--------|
| **Regularized LSE** (ridge) | `‖Aw − y‖² + λ‖w‖²` | Closed form: `w = (AᵀA + λI)⁻¹Aᵀy` |
| **Steepest Descent** (LASSO) | `‖Aw − y‖² + λ‖w‖₁` | Subgradient `2AᵀAw − 2Aᵀy + λ·sign(w)`, fixed learning rate |
| **Newton's Method** | `‖Aw − y‖²` | `w ← w − H⁻¹∇`, with Hessian `H = 2AᵀA` |

Each solver prints the fitted polynomial and its total squared error, then plots the curve against the data.

## Usage

```bash
python regression.py   # reads testfile.txt (one "x,y" per line), then prompts for n and λ
```

## Files

- `regression.py` — Matrix class and all three solvers (~237 lines)
