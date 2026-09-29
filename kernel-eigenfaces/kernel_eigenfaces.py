import os
import re
from functools import wraps
from typing import Tuple, Callable
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

try :
    import cupy as xp
    CuPy = True
    print("Using CuPy")
except ImportError :
    xp = np
    CuPy = False
    print("Using NumPy only")


CLASSES = 15
DIMENSION = 25
# SIZE = (29, 24)
SIZE = (50, 50)
# SIZE = (195, 231)
CHOICES = 10


def gpu_accelerated(func):
    """Decorator to move the data to and back from the GPU if CuPy is used."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        device_args = []
        device_kwargs = {}
        if CuPy:
            for arg in args:
                if isinstance(arg, np.ndarray):
                    arg = xp.array(arg)
                device_args.append(arg)
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    value = xp.array(value)
                device_kwargs[key] = value
        else:
            device_args = args
            device_kwargs = kwargs

        result = func(*device_args, **device_kwargs)

        if CuPy:
            if isinstance(result, xp.ndarray):
                result = xp.asnumpy(result)
            elif isinstance(result, tuple):
                result = tuple(xp.asnumpy(res) if isinstance(res, xp.ndarray) else res for res in result)
        
        return result
    return wrapper
                    
def norm_eigenvectors(eig_vectors: xp.ndarray) -> xp.ndarray:
    """Normalize the eigenvectors."""
    norm = xp.linalg.norm(eig_vectors, axis=0)
    return eig_vectors / norm

def factorize_and_reduce(S: xp.ndarray, dimension: int=None) -> Tuple[xp.ndarray, xp.ndarray]:
    """Compute and select the eigenvectors. S should be in proper device."""
    eig_vals, eig_vectors = xp.linalg.eigh(S)
    eig_vals, eig_vectors = eig_vals.real, eig_vectors.real
    sorted_idx = xp.argsort(-eig_vals)
    if dimension is None:
        sorted_idx = sorted_idx[eig_vectors[sorted_idx] > 0]
    else:
        sorted_idx = sorted_idx[:dimension]

    return eig_vals[sorted_idx], eig_vectors[:, sorted_idx]

def PCA(X: xp.ndarray, dimension: int=None) -> xp.ndarray:
    """
    Perform PCA on the input data X to reduce the dimension to the specified value.
    Returns:
        xp.ndarray: The reduced eigen matrix.
    """
    n = X.shape[0]
    num_feat = X.shape[1]

    # Compute the covariance matrix
    Mu = xp.mean(X, axis=0)
    X_center = X - Mu
    Cov = X_center @ X_center.T if n < num_feat else X_center.T @ X_center

    # Factorization and reduction
    eig_vals, eig_vectors = factorize_and_reduce(Cov, dimension)
    if n < num_feat:
        eig_vectors = X_center.T @ eig_vectors

    return norm_eigenvectors(eig_vectors)

def LDA(X: xp.ndarray, y: xp.ndarray, num_classes: int, dimension: int=None) -> xp.ndarray:
    """
    Perform LDA on the input data X to reduce the dimension to the specified value.
    Returns:
        xp.ndarray: The reduced fisher matrix.
    """
    num_feat = X.shape[1]
    Mu = xp.mean(X, axis=0)
    scatter_within = xp.zeros((num_feat, num_feat))
    scatter_between = xp.zeros((num_feat, num_feat))

    # Compute the scatter matrices
    for category in range(num_classes):
        X_cat = X[y == category]
        Mu_cat = xp.mean(X_cat, axis=0)
        dif_within = X_cat - Mu_cat
        dif_between = Mu_cat - Mu

        scatter_within += dif_within.T @ dif_within
        scatter_between += X_cat.shape[0] * dif_between.T @ dif_between

    # Factorization and reduction
    S = xp.linalg.pinv(scatter_within) @ scatter_between
    eig_vals, eig_vectors = factorize_and_reduce(S, dimension)

    return norm_eigenvectors(eig_vectors)

def linear_kernel(X: xp.ndarray, Y: xp.ndarray=None) -> xp.ndarray:
    """X and Y should be in the proper device."""
    Y = X if Y is None else Y
    return X @ Y.T

def RBF_kernel(X: xp.ndarray, Y: xp.ndarray=None, gamma: float=1e-7) -> xp.ndarray:
    """X and Y should be in the proper device."""
    Y = X if Y is None else Y
    X_sq = xp.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_sq = xp.sum(Y ** 2, axis=1).reshape(1, -1)
    return xp.exp(-gamma * (X_sq - 2 * X @ Y.T + Y_sq))

def get_kernel(kernel: str, X: xp.ndarray, Y: xp.ndarray=None, gamma: float=1e-7) -> xp.ndarray:
    """Get the kernel matrix."""
    if kernel == 'linear':
        return linear_kernel(X, Y)
    elif kernel == 'RBF':
        return RBF_kernel(X, Y, gamma=gamma)
    else:
        raise ValueError("Invalid kernel type")

def kernel_PCA(X: xp.ndarray, kernel: str, gamma: float=1e-7, dimension: int=None) -> xp.ndarray:
    """Perform PCA on the kernel matrix."""
    n = X.shape[0]
    kernel = get_kernel(kernel, X, gamma=gamma)

    # Centering the kernel matrix
    ones = xp.ones((n, n)) / n
    kernel = kernel - ones @ kernel - kernel @ ones + ones @ kernel @ ones
    _, eig_vectors = factorize_and_reduce(kernel, dimension)
    return norm_eigenvectors(eig_vectors), kernel

def kernel_LDA(X: xp.ndarray, y: xp.ndarray, num_classes: int, kernel: str, gamma: float=1e-7,
               dimension: int=None) -> xp.ndarray:
    """Perform LDA on the kernel matrix."""
    n = X.shape[0]
    kernel = get_kernel(kernel, X, gamma=gamma)
    Mu = xp.mean(kernel, axis=0)
    scatter_within = xp.zeros((n, n))
    scatter_between = xp.zeros((n, n))

    for category in range(num_classes):
        kernel_cat = kernel[y == category]
        cat_num = kernel_cat.shape[0]
        Mu_cat = xp.mean(kernel_cat, axis=0)
        dif_between = Mu_cat - Mu

        scatter_within += (kernel_cat.T @ (xp.identity(cat_num) - 1.0 / cat_num)) @ kernel_cat
        scatter_between += cat_num * dif_between.T @ dif_between

    S = xp.linalg.pinv(scatter_within) @ scatter_between
    eig_vals, eig_vectors = factorize_and_reduce(S, dimension)

    return norm_eigenvectors(eig_vectors), kernel
    

def read_data(dir_path: str, size: Tuple[int, int]):
    """
    Read the images and labels from the specified directory and resize them to the specified size.
    Returns:
        Tuple[np.ndarray, np.ndarray]: The images and labels.
    """
    files = os.listdir(dir_path)
    n = len(files)
    images = np.zeros((n, size[0] * size[1]))
    labels = np.zeros(n, dtype=int)

    for i, file in enumerate(files):
        img = Image.open(os.path.join(dir_path, file)).resize(size, Image.Resampling.LANCZOS)
        images[i] = np.array(img).flatten()
        labels[i] = int(re.search(r'subject(\d+)', file).group(1)) - 1

    return images, labels

def plot_eigen_fisher_faces(W: np.ndarray, size: Tuple[int, int], type: str, num_faces: int,
                            dir_path: str) -> None:
    """
    Plot the eigenfaces or fisherfaces.
    Args:
        W(np.ndarray): The eigenvector matrix.
        num_faces(int): The number of faces to plot.
    """
    faces = W.T[:num_faces].reshape((num_faces, *size))
    fig = plt.figure(figsize=(5, 5))
    for i, face in enumerate(faces):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.axis('off')
        ax.imshow(face, cmap='gray')
    
    fig.savefig(dir_path + f'first_{num_faces}_{type}_faces.png')
    plt.close(fig)

    return

def reconstruct_faces(W: np.ndarray, X: np.ndarray, size: Tuple[int, int], type: str,
                      num_choices: int, dir_path: str) -> None:
    """
    Reconstruct the faces using eigenfaces or fisherfaces.
    Args:
        W(np.ndarray): The eigenvector matrix.
        X(np.ndarray): The original faces.
        num_choices(int): The number of faces to reconstruct.
    """
    # Reconstruction
    choices = np.random.choice(X.shape[0], num_choices)
    reconstruct_image = X[choices] @ W @ W.T

    # Plot the reconstructed faces
    fig = plt.figure(2)
    for idx, choice in enumerate(choices):
        # Original image
        plt.subplot(num_choices, 2, 2 * idx + 1)
        plt.axis('off')
        plt.imshow(X[choice].reshape(*size), cmap='gray')

        # Reconstructed image
        plt.subplot(num_choices, 2, 2 * idx + 2)
        plt.axis('off')
        plt.imshow(reconstruct_image[idx].reshape(*size), cmap='gray')

    fig.savefig(dir_path + f'reconstructed_{type}_faces.png')
    plt.close(fig)

    return

def reconstruction(data_path: str, result_path: str) -> None:
    # Read the data
    images, labels = read_data(data_path, SIZE)

    # Perform PCA and LDA
    accelerated_PCA = gpu_accelerated(PCA)
    accelerated_LDA = gpu_accelerated(LDA)
    eigenfaces = accelerated_PCA(images, DIMENSION)
    fisherfaces = accelerated_LDA(images, labels, CLASSES, DIMENSION)

    # Plot the eigenfaces and fisherfaces
    plot_eigen_fisher_faces(eigenfaces, SIZE, 'eigen', DIMENSION, result_path)
    plot_eigen_fisher_faces(fisherfaces, SIZE, 'fisher', DIMENSION, result_path)

    # Reconstruct the faces
    reconstruct_faces(eigenfaces, images, SIZE, 'eigen', CHOICES, result_path)
    reconstruct_faces(fisherfaces, images, SIZE, 'fisher', CHOICES, result_path)

    return

def prediction(train_images: xp.ndarray, train_labels: xp.ndarray, test_images: xp.ndarray,
               test_labels: xp.ndarray, W: xp.ndarray, k: int,
               mode: str='simple', train_kernel: xp.ndarray=None) -> float:
    # Projection
    if mode != 'simple':
        if mode == 'linear':
            test_images = get_kernel('linear', test_images, train_images)
        elif mode == 'RBF':
            test_images = get_kernel('RBF', test_images, train_images)
        
        train_images = train_kernel

    projected_train = train_images @ W
    projected_test = test_images @ W

    # Apply the k-NN algorithm
    correct = 0
    for i, test in enumerate(projected_test):
        distances = xp.sum((projected_train - test) ** 2, axis=1)
        nearest = train_labels[xp.argsort(distances)[:k]]
        prediction = xp.argmax(xp.bincount(nearest))
        if prediction == test_labels[i]:
            correct += 1

    return correct / len(test_labels)

def face_recognition(train_data: str, test_data: str, k: int) -> None:
    # Read the data
    train_images, train_labels = read_data(train_data, SIZE)
    test_images, test_labels = read_data(test_data, SIZE)

    # part 2
    # Perform PCA and LDA
    accelerated_PCA = gpu_accelerated(PCA)
    accelerated_LDA = gpu_accelerated(LDA)
    eigenfaces = accelerated_PCA(train_images, DIMENSION)
    fisherfaces = accelerated_LDA(train_images, train_labels, CLASSES, DIMENSION)

    # Prediction
    accelerated_prediction = gpu_accelerated(prediction)
    accuracy = accelerated_prediction(train_images, train_labels, test_images, test_labels,
                                      eigenfaces, k)
    print(f'simple PCA accuracy: {accuracy:.2f}')
    accuracy = accelerated_prediction(train_images, train_labels, test_images, test_labels,
                                      fisherfaces, k)
    print(f'simple LDA accuracy: {accuracy:.2f}')

    # part 3
    # Perform kernel PCA and LDA
    accelerated_kernel_PCA = gpu_accelerated(kernel_PCA)
    accelerated_kernel_LDA = gpu_accelerated(kernel_LDA)
    eigenfaces, eigen_kernel = accelerated_kernel_PCA(train_images, 'linear', dimension=DIMENSION)
    fisherfaces, fisher_kernel = accelerated_kernel_LDA(train_images, train_labels, CLASSES,
                                                        'linear', dimension=DIMENSION)
    accuracy = accelerated_prediction(train_images, train_labels, test_images, test_labels,
                                      eigenfaces, k, 'linear', eigen_kernel)
    print(f'linear kernel PCA accuracy: {accuracy:.2f}')
    accuracy = accelerated_prediction(train_images, train_labels, test_images, test_labels,
                                      fisherfaces, k, 'linear', fisher_kernel)
    print(f'linear kernel LDA accuracy: {accuracy:.2f}')

    eigenfaces, eigen_kernel = accelerated_kernel_PCA(train_images, 'RBF', dimension=DIMENSION)
    fisherfaces, fisher_kernel = accelerated_kernel_LDA(train_images, train_labels, CLASSES,
                                                        'RBF', dimension=DIMENSION)
    accuracy = accelerated_prediction(train_images, train_labels, test_images, test_labels,
                                      eigenfaces, k, 'RBF', eigen_kernel)
    print(f'RBF kernel PCA accuracy: {accuracy:.2f}')
    accuracy = accelerated_prediction(train_images, train_labels, test_images, test_labels,
                                      fisherfaces, k, 'RBF', fisher_kernel)
    print(f'RBF kernel LDA accuracy: {accuracy:.2f}')

    return


if __name__ == '__main__':
    # Directory path
    result_path = 'eigenfaces_result/'
    train_data = 'Yale_Face_Database/Training/'
    test_data = 'Yale_Face_Database/Testing/'
    
    try:
        os.mkdir(result_path)
    except FileExistsError: pass

    if not os.path.exists(train_data) or not os.path.exists(test_data):
        raise FileNotFoundError("The specified data path does not exist")

    # part 1
    # reconstruction(train_data, result_path)

    # part 2, 3
    for k in [3, 5, 7]:
        print(f'k = {k}')
        face_recognition(train_data, test_data, k)