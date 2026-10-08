# Sequential Estimation & Bayesian Linear Regression in C++

Online estimation and Bayesian linear regression, built on a hand-written C++ `Matrix` class. No external linear algebra libraries.

## Data Generators

- **Univariate Gaussian**: Uses the Marsaglia polar method (a variant of Box–Muller) to turn uniform samples into Gaussian ones.
- **Polynomial basis linear model**: Draws $y = \sum_i w_i x^i + \varepsilon$ with $x \sim U(-1, 1)$ and $\varepsilon \sim \mathcal{N}(0, a)$.

## Part 1: Sequential Estimator

Draws data from $\mathcal{N}(\mu, \sigma^2)$ one point at a time and updates the running mean and variance with **Welford's algorithm**. It stops once both estimates have converged.

## Part 2: Bayesian Linear Regression

Learns the weights of a polynomial model online, one data point at a time.

**Prior:**

$$w \sim \mathcal{N}(0,\ b^{-1} I)$$

**Posterior update** for each new point $(x, y)$ with design vector $\phi = \phi(x)$:

$$\Lambda_n = a^{-1} \phi \phi^\top + \Lambda_{n-1}$$

$$\mu_n = \Lambda_n^{-1} \left( a^{-1} \phi\, y + \Lambda_{n-1} \mu_{n-1} \right)$$

**Predictive distribution:**

$$\mathcal{N}\left( \phi^\top \mu,\ a + \phi^\top \Lambda^{-1} \phi \right)$$

The run stops once the predictive variance stabilizes. Snapshots of the posterior after 10 and 50 points and at the end are written to `result.csv`.

`plot.py` reads `result.csv` and draws four panels (ground truth, final predict result, after 10 points, after 50 points), each drawing the mean curve with bounds at mean ± predictive variance.

### Matrix Class

`Matrix.h` / `Matrix.cpp` provide LU-based inversion, transpose, and arithmetic operators, with move semantics to avoid unnecessary copies.

## Build & Run

```bash
make
./main 2   # sequential estimator only
./main 3   # Bayesian linear regression only
./main 1   # both
python plot.py
```

## Files

- `main.cpp` — Both parts and result export
- `Matrix.h`, `Matrix.cpp` — Matrix class
- `header.h`, `function.cpp` — Data generators and Welford estimator
- `plot.py` — Plots the regression results
- `Makefile`
