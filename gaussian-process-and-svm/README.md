# Gaussian Process Regression & SVM

Gaussian Process regression with marginal-likelihood hyperparameter tuning, and SVM classification on MNIST with grid search and a custom kernel.

## Part 1: Gaussian Process Regression

### Approach

- **Kernel**: Rational quadratic, `k(xᵢ, xⱼ) = (1 + ‖xᵢ − xⱼ‖² / (2αl²))^(−α)`
- **Noise**: `C = K + β⁻¹I` with `β = 5`
- **Prediction** at 300 test points in `[−60, 60]`:
  - `μ(x*) = k(X, x*)ᵀ C⁻¹ y`
  - `σ²(x*) = k(x*, x*) + β⁻¹ − k(X, x*)ᵀ C⁻¹ k(X, x*)`
- **Hyperparameter optimization**: Minimizes the negative log marginal likelihood `½(ln|C| + yᵀC⁻¹y + N ln 2π)` over `(α, l)`, using `scipy.optimize.minimize`.

The kernel, the posterior, and the likelihood are all written by hand in NumPy. SciPy is used only as the optimizer.

### Results

| Initial `α = l = 1` | Optimized `α, l` |
|:---:|:---:|
| ![GP default](docs/gp_default.png) | ![GP optimized](docs/gp_optimized.png) |

*Blue: predicted mean. Red band: 95% confidence interval.*

With the default hyperparameters, the model overfits: the mean chases every training point and the band swings sharply between them. With optimized hyperparameters, both the mean and the band are much smoother.

## Part 2: SVM on MNIST

Uses [libsvm](https://github.com/cjlin1/libsvm) on MNIST images of 784 features each, read from CSV.

- **Mode 1**: Compares linear, polynomial, and RBF kernels at libsvm defaults.
- **Mode 2**: Grid-searches `C`, `γ`, degree, and `coef0` for each kernel with 3-fold cross-validation, then retrains the best model and tests it.
- **Mode 3**: Builds a custom **linear + RBF** kernel by hand, then passes the precomputed Gram matrix to libsvm (`-t 4`).

### Observations

- At libsvm defaults, the polynomial kernel performed worst of the three. It has the most hyperparameters, and untuned it generalizes poorly.
- Grid search picks the model with the best cross-validation accuracy. If the search range includes very complex settings, a model that overfits can still win that comparison and then do worse on the test set. That is why the search range matters.

## Usage

```bash
python gaussian_process.py              # needs input.data
python support_vector_machine.py 1|2|3  # needs X_train.csv, Y_train.csv, X_test.csv, Y_test.csv
```

## Files

- `gaussian_process.py` — GP regression and hyperparameter optimization
- `support_vector_machine.py` — Kernel comparison, grid search, and custom kernel
