# Naive Bayes Classifier & Beta-Binomial Conjugation

Two probabilistic models in pure Python. No NumPy or ML libraries.

## Part 1: Naive Bayes on MNIST

Classifies handwritten digits (0–9) using two likelihood models. The MNIST `idx` binary files are parsed by hand from raw bytes.

### Approach

- **Discrete mode**: Divides pixel intensities into 32 bins. Each class gets a per-pixel histogram, smoothed with a pseudocount so that no bin has zero probability.
- **Continuous mode**: Models each pixel as a per-class Gaussian. Mean and variance are computed in a single pass with **Welford's online algorithm**, and a constant is added to the standard deviation so that pixels with zero variance stay stable.
- All probabilities are kept as negative log values, so they don't underflow when multiplied over 784 pixels.

After testing, the program prints each class's "imagination": a 28×28 binary image of the pixels the model expects to be dark for that digit. It also prints the overall error rate.

### Files

- `naive_bayes.py` — Discrete and continuous classifiers (chosen at runtime)

## Part 2: Beta-Binomial Conjugation

Online Bayesian learning of a coin's bias. Each line of `testfile.txt` is a sequence of 0/1 outcomes. After reading each line, the program prints the binomial likelihood and updates the Beta prior `(a, b)` into the posterior `(a + m, b + N − m)`. That posterior then serves as the prior for the next line.

### Files

- `beta_binomial.py`

## Usage

```bash
python naive_bayes.py     # needs the four MNIST idx files in the working directory
python beta_binomial.py   # needs testfile.txt
```
