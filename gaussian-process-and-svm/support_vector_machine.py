import sys
import numpy as np
import time
from libsvm.svmutil import *

kernel = {
    'Linear': 0, 
    'Polynomial': 1, 
    'RBF': 2,
    'Precomputed': 4
}

def readCSV(filename: str, dtype: np.dtype=np.float64) -> np.ndarray:
    """
    Read a CSV file and return its content as a numpy array

    Args:
        filename (str): The path of the CSV file
        dtype (np.dtype): The data type of the elements of the array

    """
    return np.genfromtxt(filename, delimiter=',', dtype=dtype)

def title_and_params(kernel_type: str, degree: float=None,
                      gamma: float=None, coef0: float=None, cost: float=None,
                      cross_validation: bool=None) -> tuple:
    """
    Generate the title and the parameters for the SVM model

    Args:
        kernel_type (str): The type of the kernel function
        degree (float): Default is None
        gamma (float): Default is None
        coef0 (float): Default is None
        cost (float): Default is None   
        cross_validation (bool): Cross validation mode. Default is None

    Returns:
        str: The title of the model
        svm_parameter: The parameters of the model
    """
    title = f'{kernel_type} kernel'
    training_options = f'-t {kernel[kernel_type]}'

    options = (('-d', 'degree', degree),
               ('-g', 'gamma', gamma),
               ('-r', 'coef0', coef0),
               ('-c', 'cost', cost))
    
    for opt, name, value in options:
        if value:
            training_options += f' {opt} {value}'
            title += f', {name}={value}'

    if cross_validation:
        training_options += ' -v 3'

    training_options += ' -q'

    return title, svm_parameter(training_options)
    

def train_and_predict(prob: svm_problem, Y: np.ndarray, X: np.ndarray,
                      title: str, svm_params: svm_parameter) -> tuple:
    """
    Predict the result of the model and print the accuracy
    
    Args:
        prob (svm_problem): Combination of the train data
        Y (np.ndarray): The labels of the test data
        X (np.ndarray): The features of the test data
        title (str): The title for the output
        svm_params (svm_parameter): The parameters for the model

    Returns:
        float: The accuracy of the model
        float: The elapsed time of the model
    """
    print(title + ':')

    start_time = time.time()
    model = svm_train(prob, svm_params)
    end_time = time.time()
    _, p_acc, _ = svm_predict(Y, X, model)
    train_time = end_time - start_time

    print(f'Elapsed time = {train_time:.2f} seconds\n')
    return p_acc[0], train_time

def grid_search(prob: svm_problem, Y: np.ndarray, X: np.ndarray,
                kernel_type: str, degree: list=[None], gamma: list=[None],
                coef0: list=[None], cost: list=[None]) -> tuple:
    """
    Predict the result of the model and print the accuracy
    
    Args:
        prob (svm_problem): Combination of the train data
        Y (np.ndarray): The labels of the test data
        X (np.ndarray): The features of the test data
        kernel_type (str): The type of the kernel function
        degree (float): List of degree. Default is [None]
        gamma (float): List of gamma. Default is [None]
        coef0 (float): List of coef0. Default is [None]
        cost (float): List of cost. Default is [None]

    Returns:
        float: The accuracy of the best model
        str: The title of the best model
        svm_parameter: The parameters of the best model
    """

    max_acc = 0
    for d in degree:
        for g in gamma:
            for r in coef0:
                for c in cost:
                    title, params = title_and_params(kernel_type, d, g, r, c,
                                                     True)
                    print(f'Grid search: {title}')                                                     
                    acc = svm_train(prob, params)
                    print('')
                    if acc > max_acc:
                        max_acc = acc
                        best_title = title
                        best_params = params
    
    return max_acc, best_title, best_params

def linear_kernel(X_lt: np.ndarray, X_rt: np.ndarray=None) -> np.ndarray:
    """
    Compute the linear kernel of two matrices

    Args:
        X_lt (np.ndarray): The left matrix
        X_rt (np.ndarray): The right matrix. If None, set to X_lt. Default is None

    Returns:
        np.ndarray: The result of the kernel
    """
    if X_rt is None:
        X_rt = X_lt
    return X_lt @ X_rt.transpose()

def RBF_kernel(X_lt: np.ndarray, X_rt: np.ndarray=None,
               gamma: float=None) -> np.ndarray:
    """
    Compute the RBF kernel of two matrices
    
    Args:
        X_lt (np.ndarray): The left matrix
        X_rt (np.ndarray): The right matrix. If None, set to X_lt. Default is None
        gamma (float): If None, set to 1 / num_train. Default is None
    Returns:
        np.ndarray: The result of the kernel
    """
    if X_rt is None:
        X_rt = X_lt
    if gamma is None:
        gamma = 1 / len(X_lt)

    X_lt_sq = np.sum(X_lt ** 2, axis=1).reshape(-1, 1)
    X_rt_sq = np.sum(X_rt ** 2, axis=1)
    return np.exp(-gamma * (X_lt_sq + X_rt_sq - 2 * X_lt @ X_rt.transpose()))


usage = """
Main function
usage: python3 support_vector_machine.py [-mode]
mode:
    1. Comparing three types of kernel
    2. Grid search
    3. Combine Linear and RBF kernel
"""


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(usage)
        exit()

    # Read the data
    X_train = readCSV('X_train.csv')
    X_test = readCSV('X_test.csv')
    Y_train = readCSV('Y_train.csv', np.int32)
    Y_test = readCSV('Y_test.csv', np.int32)
    num_train = len(Y_train)
    num_test = len(Y_test)

    # Convert to svm problem
    prob_train = svm_problem(Y_train, X_train)

    # Choose the mode
    mode = sys.argv[1]
    if mode == '1':
        # Use different kernel functions (linear, polynomial, RBF)
        train_and_predict(prob_train, Y_test, X_test,
                        *title_and_params('Linear'))
        train_and_predict(prob_train, Y_test, X_test,
                        *title_and_params('Polynomial'))
        train_and_predict(prob_train, Y_test, X_test,
                        *title_and_params('RBF'))
    elif mode == '2':
        # Grid search for the best hyperparameters
        degree = [1, 2]
        gamma = [1 / num_train, 1e-3, 1e-2, 0.1, 1]
        coef0 = [0, 0.5, 1, 5, 10]
        cost = [0.1, 1, 10, 100]
        best_model = [grid_search(prob_train, Y_test, X_test, 'Linear',
                                cost=cost),
                    grid_search(prob_train, Y_test, X_test, 'Polynomial',
                                degree=degree, gamma=gamma, coef0= coef0, cost=cost),
                    grid_search(prob_train, Y_test, X_test, 'RBF', gamma=gamma,
                                cost=cost)]
        best_model.sort()
        best_model[-1][-1].cross_validation = False
        print('Best model after grid search:')
        train_and_predict(prob_train, Y_test, X_test, *best_model[-1][1:])
    elif mode == '3':
        # Combine Linear and RBF kernel
        start_time = time.time()
        combined_kernel = linear_kernel(X_train) + RBF_kernel(X_train)
        combined_kernel = np.hstack((np.arange(1, num_train + 1).reshape(-1, 1),
                                    combined_kernel))
        combined_model = svm_train(Y_train, combined_kernel,
                                f'-t {kernel["Precomputed"]} -q')
        end_time = time.time()
        
        test_kernel = linear_kernel(X_test, X_train) + RBF_kernel(X_test, X_train)
        test_kernel = np.hstack((np.arange(1, num_test + 1).reshape(-1, 1),
                                test_kernel))
        print('Combined kernel: Linear + RBF')
        svm_predict(Y_test, test_kernel, combined_model)
        print(f'Train time = {end_time - start_time:.2f} seconds')