from typing import Callable
from scipy.spatial import distance
from PIL import Image
from matplotlib import pyplot as plt
import io
import time
from tqdm import tqdm
import numpy as np


# Set the random seed for reproducibility
np.random.seed(0)


class Kernel_K_Means:
    def __init__(self, gram_matrix: np.ndarray,
                 compare: Callable[[float, float], bool]=lambda x, y: x > y):
        """
        Args:
            gram_matrix (np.ndarray): Gram matrix
            compare (Callable[[float, float], bool]): Comparison function for optimization based on kernel type
        """
        self.gram_matrix = gram_matrix
        self.compare = compare
        self.data_num = gram_matrix.shape[0]
        self.cluster_num = 0
        self.cluster_sizes = None
        self.labels = None

    def __del__(self):
        """Destructor to ensure all np.ndarray objects are deleted from the memory"""
        self.gram_matrix = None
        self.cluster_sizes = None
        self.labels = None

    def _reset(self, cluster_num: int=0):
        """Reset the cluster sizes and labels"""
        self.cluster_num = cluster_num
        self.cluster_sizes = None
        self.labels = None

    def _get_opt(self, data: np.ndarray) -> tuple:
        """
        Returns:
            tuple: optimal index, optimal value
        """
        opt_idx = None
        for idx, val in np.ndenumerate(data):
            if opt_idx is None or self.compare(val, data[opt_idx]):
                opt_idx = idx

        return opt_idx, data[opt_idx]

    def _get_cluster_kernel_sum(self):
        """Calculate the sum of the kernel values of each cluster"""
        cluster_kernel_sum = np.zeros(self.cluster_num)
        for c in range(self.cluster_num):
            cluster_kernel_sum[c] = self.gram_matrix[self.labels == c][:, self.labels == c].sum()
            cluster_kernel_sum[c] /= (self.cluster_sizes[c] ** 2)

        return cluster_kernel_sum

    def init_cluster(self, cluster_num: int, mode: str):
        """Initialize the cluster centroids and labels"""
        self._reset(cluster_num)
        if mode == 'random':
            self.labels = np.random.randint(0, cluster_num, self.data_num)
        elif mode == 'kmeans++':
            # Randomly choose the centroid of the first cluster
            centroids = np.full(cluster_num, -1)
            centroids[0] = np.random.randint(0, self.data_num)
            self.labels = np.zeros(self.data_num, dtype=int)
            
            # Choose the centroid of the remaining clusters
            similarity = np.array([self.gram_matrix[i, centroids[0]] for i in range(self.data_num)])
            for c in range(1, cluster_num):
                # Compute the similarity of each pixel to the nearest centroid
                for i in range(self.data_num):
                    if self.compare(self.gram_matrix[i, centroids[c-1]], similarity[i]):
                        similarity[i] = self.gram_matrix[i, centroids[c-1]]
                        self.labels[i] = c

                # Choose the centroid randomly based on the similarity 
                prob = similarity / similarity.sum()
                # REVIEW: Check if the probability distribution is correct
                centroids[c] = np.random.choice(self.data_num, p=prob)
                self.labels[centroids[c]] = c

        # Count the number of pixels in each cluster
        self.cluster_sizes = np.zeros(cluster_num, dtype=int)
        for label in self.labels:
            self.cluster_sizes[label] += 1

        return

    def _ME_step(self):
        """Perform the Maximization (minimization) and Expectation steps of the kernel k-means algorithm"""
        # Maximization (minimization) step
        cluster_kernel_sum = self._get_cluster_kernel_sum()

        # Expectation step
        new_labels = np.full(self.data_num, -1)
        for i in range(self.data_num):
            for c in range(self.cluster_num):
                inertia = -2 * self.gram_matrix[i, self.labels == c].sum() / self.cluster_sizes[c]
                inertia += cluster_kernel_sum[c]

                if new_labels[i] == -1 or self.compare(inertia, opt_inertia):
                    new_labels[i] = c
                    opt_inertia = inertia

        # Update the labels
        changes = 0
        for i in range(self.data_num):
            if new_labels[i] != self.labels[i]:
                self.cluster_sizes[self.labels[i]] -= 1
                self.cluster_sizes[new_labels[i]] += 1
                self.labels[i] = new_labels[i]
                changes += 1
        
        return changes

    def run_algorithm(self, gif_path: str, error: int=5, max_iter: int=100):
        """Perform the kernel k-means algorithm"""
        # Save the frame for each iteration
        frames = []
        for iter in tqdm(range(max_iter), desc='Running algorithm'):
            # ME step
            changes = self._ME_step()

            # Save the label image
            save_label_image(self.labels.reshape(100, 100), frames)
            
            if changes <= error: break

        # Create the GIF
        create_gif(gif_path, frames)

        return iter


class Combined_Kernel_K_Means(Kernel_K_Means):
    """Kernel K-Means algorithm with combined RBF kernel"""
    def __init__(self, spatial_kernel: np.ndarray, color_kernel: np.ndarray,
                 compare: Callable[[float, float], bool]=lambda x, y: x < y):
        """
        Args:
            spatial_kernel (np.ndarray): Spatial kernel
            color_kernel (np.ndarray): Color kernel
            compare (Callable[[float, float], bool]): Comparison function for optimization based on kernel type.
                                                      The default is for the combined RBF kernel.
                                                      The larger the value, the more similar the data points are.
        """
        super().__init__(gram_matrix=spatial_kernel, compare=compare)
        self.spatial_kernel = spatial_kernel
        self.color_kernel = color_kernel

    def __del__(self):
        """Destructor to ensure all np.ndarray objects are deleted from the memory"""
        self.spatial_kernel = None
        self.color_kernel = None

    def init_cluster(self, cluster_num: int, mode: str, gamma_s: float=1.0, gamma_c: float=1.0):
        """Initialize the cluster centroids and labels"""
        # Compute the combined RBF kernel
        self.gram_matrix = combined_RBF_kernel(self.spatial_kernel, self.color_kernel, gamma_s, gamma_c)
        super().init_cluster(cluster_num, mode)
        

def open_image(file):
    img = Image.open(file)
    return np.array(img)

def save_label_image(labels: np.ndarray, frames: list):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    cax = ax.imshow(labels, cmap='PiYG')
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    frames.append(Image.open(buf))

    plt.close(fig)
    return

def create_gif(title: str, frames: list=None):
    frames[-1].save(f'{title}.gif', format='GIF', save_all=True, append_images=frames[:], duration=100, loop=0)
    return

def compute_spatial_distance(image: np.ndarray) -> np.ndarray:
    """Compute the spatial distance array of an image"""
    height, width, _ = image.shape
    coordinates = np.array([[i, j] for i in range(height) for j in range(width)])
    return distance.cdist(coordinates, coordinates, 'sqeuclidean')

def compute_color_distance(image: np.ndarray) -> np.ndarray:
    """Compute the color distance array of an image"""
    return distance.cdist(image.reshape(-1, 3), image.reshape(-1, 3), 'sqeuclidean')


def combined_RBF_kernel(spatial_distance: np.ndarray, color_distance: np.ndarray,
                        gamma_s: float=1.0, gamma_c: float=1.0) -> np.ndarray:
    """
    Compute the combined RBF kernel of two matrices X and Y

    Returns:
        np.ndarray: Gram matrix
    """
    return np.exp(-gamma_s * spatial_distance) * np.exp(-gamma_c * color_distance)


if __name__ == '__main__':
    # Hyperparameters
    clusters = [2, 3, 4, 5]
    initial = ['random', 'kmeans++']
    gamma_s = [1e-2, 1e-3, 1e-4]
    gamma_c = [1e-2, 1e-3, 1e-4]

    # Image path
    image_paths = ['./image1.png', './image2.png']
    save_gif_paths = ['./KKM_gif/image1/', './KKM_gif/image2/']
    for img_path, save_gif_path in zip(image_paths, save_gif_paths):
        for gs in gamma_s:
            for gc in gamma_c:
                # Load the image
                image = open_image(img_path)
                
                # Initialize the kernel k-means algorithm
                spatial_kernel = compute_spatial_distance(image)
                color_kernel = compute_color_distance(image)
                kernel_k_means = Combined_Kernel_K_Means(spatial_kernel, color_kernel)
                for cluster in clusters:
                    for init in initial:
                        print(f'Processing {init} kernel k-means with {cluster} clusters...')
                        start_time = time.time()
                        kernel_k_means.init_cluster(cluster, init, gs, gc)
                        iter = kernel_k_means.run_algorithm(f'{save_gif_path}{init}_{cluster}_clusters_{gs}_{gc}')
                        end_time = time.time()
                        print(f'Number of iterations: {iter + 1}')
                        print(f'Elapsed time: {end_time - start_time:.2f} seconds\n')
        
    