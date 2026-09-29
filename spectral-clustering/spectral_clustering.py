from __future__ import annotations
from typing import Callable, Union
from types import ModuleType
from scipy.spatial import distance
from PIL import Image
from matplotlib import pyplot as plt
import io
import time
from tqdm import tqdm
import numpy as np
import gc

from kernel_kmeans import (
    Kernel_K_Means,
    compute_spatial_distance,
    compute_color_distance,
    combined_RBF_kernel,
    open_image
)

try :
    import cupy as cp
    has_gpu = cp.cuda.runtime.getDeviceCount() > 0
except ImportError:
    has_gpu = False

print("CuPy is installed and GPU is available") if has_gpu else print("CuPy is NOT installed or GPU is NOT available")


# Set the random seed for reproducibility
np.random.seed(0)


class Spectral_Clustering:
    def __init__(self, image: Image):
        """"""
        self.data = open_image(image)
        self.data_num = self.data.shape[0]
        self.spatial_distance = compute_spatial_distance(self.data)
        self.color_distance = compute_color_distance(self.data)
        self.L = None
        self.eigen_idx = None
        self.eigenvectors = None

    def __del__(self):
        """Destructor to ensure all np.ndarray objects are deleted from the memory"""
        self.spatial_distance = None
        self.color_distance = None
        self.L = None
        # self.labels = None
        self.eigen_idx = None
        self.eigenvectors = None

        gc.collect()
        return

    def _reset(self):
        """Reset laplacian"""
        self.L = None
        # self.cluster_num = 0
        # self.labels = None
        self.eigen_idx = None
        self.eigenvectors = None

        gc.collect()
        return

    def _compute_laplacian(self, gamma_s: float, gamma_c: float,
                           method: Callable[[np.ndarray, np.ndarray, ModuleType], np.ndarray]):
        """Compute the similarity matrix"""
        weight = combined_RBF_kernel(self.spatial_distance, self.color_distance, gamma_s, gamma_c)

        if has_gpu:
            weight = cp.asarray(weight)
            xp = cp
        else:
            xp = np
        diagonal = xp.diag(xp.sum(weight, axis=1))

        self.L = method(diagonal, weight, xp)

        return 

    def _compute_eigenvectors(self):
        """Compute the eigenvectors of the normalized Laplacian"""
        start_time = time.time()
        if has_gpu:
            print("Using GPU for eigenvector computation...")
            eigenvalues, eigenvectors = cp.linalg.eigh(self.L)
            eigenvalues = cp.asnumpy(eigenvalues)
            self.eigenvectors = cp.asnumpy(eigenvectors)
        else:
            print("Using CPU for eigenvector computation...")
            eigenvalues, self.eigenvectors = np.linalg.eigh(self.L)

        print(f"Time taken to compute eigenvectors: {time.time() - start_time:.2f} seconds\n")
        self.eigen_idx = np.argsort(eigenvalues)
        return

    def set_gamma(self, gamma_s: float, gamma_c: float,
                  method: Callable[[np.ndarray, np.ndarray, ModuleType], np.ndarray]):
        """Set the gamma values for the RBF kernel"""
        self._reset()
        self._compute_laplacian(gamma_s, gamma_c, method)
        self._compute_eigenvectors()

        return

    def run_spectral_clustering(self, cluster_num: int, save_path_pre: str, kkm_init: str='random',
                                method: str='normalized'):
        """Run the spectral clustering algorithm"""
        # Compute kernel based on normalized eigenspace
        U = self.eigenvectors[:, self.eigen_idx[1:cluster_num + 1]]
        if method == 'normalized':
            L2_norm = np.linalg.norm(U, axis=1, keepdims=True)
            U = U / L2_norm
        
        kernel = distance.cdist(U, U, 'sqeuclidean')

        # Perform kernel K-means clustering
        kkm = Kernel_K_Means(kernel)
        kkm.init_cluster(cluster_num, kkm_init)
        iter = kkm.run_algorithm(save_path_pre)

        # Plot eigenspace
        if cluster_num == 2:
            plot2D_eigenspace(U, kkm.labels, '2D Eigenspace', f'{save_path_pre}_2D_eigenspace.png')
        elif cluster_num == 3:
            plot3D_eigenspace(U, kkm.labels, '3D Eigenspace', f'{save_path_pre}_3D_eigenspace.png')

        return


def _ratio_cut(D: Union[np.ndarray, cp.ndarray], W: Union[np.ndarray, cp.ndarray],
               xp: ModuleType) -> Union[np.ndarray, cp.ndarray]:
    """Compute the ratio cut Laplacian"""
    return D - W

def _normalized_laplacian(D: Union[np.ndarray, cp.ndarray], W: Union[np.ndarray, cp.ndarray],
                          xp: ModuleType) -> Union[np.ndarray, cp.ndarray]:
    """Compute the normalized Laplacian"""
    D = xp.linalg.inv(xp.sqrt(D))
    return xp.identity(D.shape[0]) - D @ W @ D

def plot2D_eigenspace(U: np.ndarray, labels: np.ndarray, title: str, save_path: str):
    fig = plt.figure()
    x = U[:, 0]
    y = U[:, 1]
    plt.xlabel('Eigenvector 1')
    plt.ylabel('Eigenvector 2')
    plt.title(title)

    colormap = plt.colormaps.get_cmap('viridis')
    colors = colormap(np.linspace(0, 1, 2))

    for i in range(2):
        plt.scatter(x[labels == i], y[labels == i], color=colors[i], label=f'Cluster {i}')

    plt.legend()
    plt.savefig(save_path)
    plt.close(fig)
    return

def plot3D_eigenspace(U: np.ndarray, labels: np.ndarray, title: str, save_path: str):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = U[:, 0]
    y = U[:, 1]
    z = U[:, 2]
    ax.set_xlabel('Eigenvector 1')
    ax.set_ylabel('Eigenvector 2')
    ax.set_zlabel('Eigenvector 3')
    plt.title(title)

    colormap = plt.colormaps.get_cmap('viridis')
    colors = colormap(np.linspace(0, 1, 3))

    for i in range(3):
        ax.scatter(x[labels == i], y[labels == i], z[labels == i], color=colors[i], label=f'Cluster {i}')
    
    plt.legend()
    plt.savefig(save_path)
    plt.close(fig)
    return


if __name__ == "__main__":
    methods = ['normalized', 'ratio_cut']
    clusters = [2, 3, 4]
    initial = ['random', 'kmeans++']
    gamma_s = [1e-3, 1e-4]
    gamma_c = [1e-3, 1e-4]

    image_paths = ['./image1.png', './image2.png']
    save_gif_paths = ['./spectral_gif/image1/', './spectral_gif/image2/']
    for img_path, save_gif_path in zip(image_paths, save_gif_paths):
        spectral_clustering = Spectral_Clustering(img_path)
        for g_s in gamma_s:
            for g_c in gamma_c:
                for method in methods:
                    if method == 'normalized':
                        spectral_clustering.set_gamma(g_s, g_c, _normalized_laplacian)
                    else:
                        spectral_clustering.set_gamma(g_s, g_c, _ratio_cut)

                    for cluster in clusters:
                        for init in initial:
                            print(f'{methods} spectral clustering, gamma = ({g_s}, {g_c}), {init} initialization')
                            print(f'cluster = {cluster}, processing...')
                            start_time = time.time()
                            spectral_clustering.run_spectral_clustering(cluster, f'{save_gif_path}{init}_{cluster}_clusters_{g_s}_{g_c}', init)
                            end_time = time.time()
                            print(f'Elapsed time: {end_time - start_time:.2f} seconds\n')

        del spectral_clustering
        gc.collect()