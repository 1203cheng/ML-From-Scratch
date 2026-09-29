# Sequential Estimation & Bayesian Linear Regression in C++

Online estimation and Bayesian linear regression, built on a hand-written C++ `Matrix` class. No external linear algebra libraries.

## Data Generators

- **Univariate Gaussian**: Uses the Marsaglia polar method (a variant of Box–Muller) to turn uniform samples into Gaussian ones.
- **Polynomial basis linear model**: Draws `y = Σ wᵢxⁱ + ε` with `x ~ U(−1, 1)` and `ε ~ N(0, a)`.

## Part 1: Sequential Estimator

Draws data from `N(μ, σ²)` one point at a time and updates the running mean and variance with **Welford's algorithm**. It stops once both estimates have converged.

## Part 2: Bayesian Linear Regression

Learns the weights of a polynomial model online, one data point at a time.

- **Prior**: `w ~ N(0, b⁻¹I)`
- **Posterior update** for each new point `(x, y)` with design vector `φ(x)`:
  - `Λₙ = a⁻¹ φφᵀ + Λₙ₋₁`
  - `μₙ = Λₙ⁻¹ (a⁻¹ φ y + Λₙ₋₁ μₙ₋₁)`
- **Predictive distribution**: `N(φᵀμ, a + φᵀΛ⁻¹φ)`

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
