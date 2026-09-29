#  Adding SNE to the t-SNE implementation
#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import pylab
from PIL import Image
from matplotlib import pyplot as plt
from io import BytesIO
import os
import sys

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def pairwise_dist(X=np.array([])):
    """
        Compute the pairwise distance matrix for a set of vectors X.
    """

    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return D


def gaussian_affinities(X=np.array([])):
    """
    Compute the Gaussian affinities for a set of vectors X.
    """

    D = pairwise_dist(X)
    num = np.exp(-D)
    Q = num / np.sum(num)

    return Q, num


def t_distribution_affinities(X=np.array([])):
    """
    Compute the t-distribution affinities for a set of vectors X.
    """

    num = 1. / (1. + pairwise_dist(X))
    n = X.shape[0]
    num[range(n), range(n)] = 0.
    Q = num / np.sum(num)

    return Q, num


def tsne_sne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0,
             affinities=gaussian_affinities, labels=None, mode='SNE'):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    if labels is None:
        raise ValueError('Labels are required for visualization')

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    frames = []
    for iter in range(max_iter):

        # Compute pairwise affinities
        Q, num = affinities(Y)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

            if (iter + 1) % 20 == 0:
                save_label_images(Y, labels, 
                                  f'{mode}, perplexity = {perplexity}, iteration = {iter + 1}',
                                  frames)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Visualize the result 
    create_gif(result_path + f'{mode}_perplexity_{perplexity}.gif'.replace('-', '_'), frames)
    plot_similarity(P, Q, f'{mode}, perplexity = {perplexity}',
                   result_path + f'{mode}_perplexity_{perplexity}.png'.replace('-', '_'))

    # Return solution
    return Y


def save_label_images(Y, labels, title, frames):
    """
    Visualize the t-SNE or SNE results
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='viridis')
    ax.set_title(title)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf))

    return


def create_gif(filename, frames):
    """
    Create a GIF from the frames
    """
    frames[-1].save(filename, format='GIF', save_all=True, append_images=frames[:], duration=100,
                    loop=0)
    return


def plot_similarity(P, Q, title, filename):
    """
    Plot the similarity matrix
    """

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=16)

    cmap = plt.get_cmap('coolwarm')

    axs[0].set_title('High-D Similarity')
    axs[0].hist(P.flatten(), bins=40, log=True, color=cmap(0.2))

    axs[1].set_title('Low-D Similarity')
    axs[1].hist(Q.flatten(), bins=40, log=True, color=cmap(0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    plt.close(fig)

    return


def command_line_error():
    print("Usage: python3 tsne_sne.py [t-SNE|SNE] <perplexity>")
    sys.exit(1)


if __name__ == "__main__":
    """
    Main function
    usage: python3 tsne_sne.py [t-SNE|SNE] <perplexity>
    """

    # Parse command line arguments
    if len(sys.argv) < 3:
        command_line_error()

    mode = sys.argv[1]
    perplexity = float(sys.argv[2])
    if mode == 't-SNE':
        affinities = t_distribution_affinities
    elif mode == 'SNE':
        affinities = gaussian_affinities
    else:
        command_line_error()

    # Create the result path
    result_path = 't_sne_result/'
    try:
        os.mkdir(result_path)
    except FileExistsError: pass

    # Run the t-SNE or SNE
    print(f"Run {mode}, perplexity={perplexity} to on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("tsne_python/mnist2500_X.txt")
    labels = np.loadtxt("tsne_python/mnist2500_labels.txt")
    Y = tsne_sne(X, 2, 50, perplexity=perplexity, affinities=affinities, labels=labels, mode=mode)
    # pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    # pylab.show()
