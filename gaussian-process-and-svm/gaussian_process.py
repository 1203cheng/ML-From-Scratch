import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def rational_quadratic_kernel(X: np.ndarray,
                              Y: np.ndarray=None,
                              alpha: float=1.0,
                              length: float=1.0) -> np.ndarray:
    """
    Calculate rational quadratic kernel.

    Args:
        X (np.ndarray): shape (num, features).
        Y (np.ndarray): shape (num, features). If None, set to X. Default to None.
        alpha (float): scale-mixture alpha. Default to 1.0.
        length (float): length-scale. Default to 1.0.

    Returns:
        np.ndarray: gram matrix
     """
    if Y is None: Y = X

    square_dist = np.sum(X ** 2, 1).reshape(-1, 1) + np.sum(Y ** 2, 1) - (2 * (X @ Y.T))
    return (1 + square_dist / (2 * alpha * (length ** 2))) ** (-alpha)

def gaussian_process(X_train: np.ndarray,
                     Y_train: np.ndarray,
                     beta: float=5.0,
                     alpha: float=1.0,
                     length: float=1.0,
                     title: str='Gaussian process') -> None:
    """
    Apply gaussian process regression to predict the distribution and visualize it.

    Args:
        X_train (np.ndarray): shape (num, feature)
        Y_train (np.ndarray): shape (num, feature)
        beta (float): inverse of noise. Default to 5.0.
        alpha (float): Params for rational quadratic kernel. Default to 1.0.
        length (float): Params for rational quadratic kernel. Default to 1.0.
        title (str): Title of graph. Default to 'Gaussian process'. 

    Returns:
        None
    """
    # Initialize parameter
    noise = 1.0 / beta

    # Calculate covariance matrix
    K = rational_quadratic_kernel(X_train, alpha=alpha, length=length)
    K_noise = np.zeros_like(K)
    np.fill_diagonal(K_noise, noise)
    C = K + K_noise

    # Generate test data
    X_test = np.linspace(-60, 60, 300).reshape(-1, 1)

    # Prediction of test data
    # Calculate parameters
    K_train_test = rational_quadratic_kernel(X_train, X_test, alpha=alpha, length=length)
    K_train_test_T = K_train_test.transpose()
    C_inverse = np.linalg.inv(C)
    kernel_test_list = [rational_quadratic_kernel(x.reshape(1, -1),
                                                  alpha=alpha,
                                                  length=length)[0, 0] for x in X_test]
    c = (np.array(kernel_test_list).reshape(X_test.shape) + noise).ravel()

    # Calculate Mean and Variance
    Mean = (K_train_test_T @ (C_inverse @ Y_train)).ravel()
    Variance = np.zeros_like(Mean)
    for i, k in enumerate(K_train_test_T):
        Variance[i] = k @ C_inverse @ k.transpose()

    Variance = (c - Variance)

    # Visualization
    # 95% confidence interval
    Sigma = np.sqrt(Variance)
    upper_bound = Mean + 1.96 * Sigma
    lower_bound = Mean - 1.96 * Sigma

    plt.xlim(-60, 60)
    plt.title(title)
    plt.scatter(X_train.ravel(), Y_train.ravel(), c='m', label='Training Data')
    plt.plot(X_test.ravel(), Mean, label='Mean Prediction', color='b')
    plt.fill_between(X_test.ravel(), upper_bound, lower_bound, color='r', alpha=0.5)
    plt.tight_layout()
    plt.show()

def negative_log_marginal_likelihood(theta: np.ndarray,
                                     X_train: np.ndarray,
                                     Y_train: np.ndarray,
                                     beta: float=5.0) -> float:
    """
    Function of negative logarithm marginal likelihood of y: -ln p(y|theta).

    Args:
        theta (np.ndarray): 1-D array (alpha, length)
        X_train (np.ndarray): shape (num, feature)
        Y_train (np.ndarray): shape (num, feature)
        beta (float): inverse of noise. Default to 5.0

    Returns:
        float: -ln p(y|theta)
    """
    noise = 1.0 / beta
    K = rational_quadratic_kernel(X_train, alpha=theta[0], length=theta[1])
    K_noise = np.zeros_like(K)
    np.fill_diagonal(K_noise, noise)
    C = K + K_noise 
    return 0.5 * (np.log(np.linalg.det(C)) + Y_train.transpose() @ np.linalg.inv(C) @ Y_train
                   + len(X_train) * np.log(2 * np.pi))


if __name__ == '__main__':
    """
    Main function
    usage: python3 gaussian_process.py
    """
    # Load data from 'input.data'
    X_train = np.loadtxt('input.data', usecols=0)
    Y_train = np.loadtxt('input.data', usecols=1)
    X_train = X_train.reshape(-1, 1)
    Y_train = Y_train.reshape(-1, 1)

    # Gaussian process
    gaussian_process(X_train, Y_train)

    # Optimize the kernel params
    theta = np.array([1.0, 1.0])
    opt = minimize(negative_log_marginal_likelihood,
                   theta,
                   args=(X_train, Y_train))

    gaussian_process(X_train, Y_train, alpha=opt.x[0], length=opt.x[1],
                     title='Optimized gaussian process')


