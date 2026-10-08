# Logistic Regression & EM Algorithm in C++

Binary classification and unsupervised digit clustering, built on a hand-written C++ `Matrix` class. No external ML or linear algebra libraries.

## Part 1: Logistic Regression

Separates two classes of 2D points drawn from Gaussians. The user enters the means and variances of each class.

### Approach

- **Gradient descent**: Follows the log-likelihood gradient $X^\top \left( t - \sigma(Xw) \right)$.
- **Newton's method**: Scales the gradient by the inverse Hessian $H = X^\top D X$, where $D = \operatorname{diag}\left( \sigma'(Xw) \right)$. If $H$ is singular, it falls back to a plain gradient step.
- Both methods add a momentum term to the update and stop when the relative change in $w$ is small enough.

Outputs the learned weights, the confusion matrix, sensitivity, and specificity. `plot.py` draws the ground truth next to each method's predictions.

## Part 2: EM Algorithm on MNIST

Clusters 60,000 binarized MNIST digits into 10 groups without labels, using a mixture of Bernoulli distributions.

### Approach

- **E-step**: Computes each image's responsibility for each cluster from the per-pixel Bernoulli parameters and the mixing weights $\lambda$.
- **M-step**: Re-estimates the Bernoulli parameters and mixing weights from those responsibilities.
- **Restart on collapse**: If any mixing weight drops to zero, the parameters are randomly re-initialized.
- **Label assignment**: After convergence, the program builds a count table of true digits against clusters. It then repeatedly matches the largest remaining (digit, cluster) pair, giving a one-to-one mapping.

Prints each cluster's "imagination" (a 28×28 binary image) every iteration. At the end it prints a confusion matrix, sensitivity, and specificity per digit, plus the overall error rate.

## Build & Run

```bash
make
./main 1   # logistic regression
./main 2   # EM algorithm (needs the MNIST training idx files)
./main 3   # both
python plot.py
```

## Files

- `main.cpp` — Both parts (~650 lines)
- `Matrix.h`, `Matrix.cpp` — Matrix class
- `RandomData.h`, `RandomData.cpp` — Gaussian data generator
- `plot.py` — Plots the logistic regression results
- `Makefile`
