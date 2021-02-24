import mykmeanssp as kmns
import numpy as np


def k_means_pp(vectors, K, d, N, MAX_ITER):
    """variables are:
    centroids_count : count hpe many centroids have been calculated
    distance : the distence from each vector to the closest centroid
    centroid_indices : ndarry containing the indices of the vectors chosen to be centroids
    centroids : ndarray containing the vectors chosen to be centroids
    """
    np.random.seed(0)  # pick the seed val to be 0
    # indices = np.arange(N)
    centroids_count = 1

    centroid_indices = np.zeros(K, dtype=np.int64)
    centroid_indices[0] = np.random.choice(N, 1)
    centroids = np.zeros((K, d), dtype=np.float64)

    # step 1 put an initial vector by random

    centroids[0] = vectors[centroid_indices[0]]
    distance = np.power(np.linalg.norm(vectors - centroids[centroids_count - 1], axis=1), 2)
    while centroids_count < K:
        new_dist = np.power(np.linalg.norm(vectors - centroids[centroids_count - 1], axis=1), 2)
        distance = np.minimum(distance, new_dist)
        # step 3 make a probability vector from the distances vector
        p = distance / distance.sum()
        # step 4 set the new centroid by taking the
        centroid_indices[centroids_count] = np.random.choice(N, 1, p=p)
        centroids[centroids_count] = vectors[centroid_indices[centroids_count]]
        centroids_count += 1
    vectors.reshape(vectors.size)
    centroids.reshape(centroids.size)
    kmns.kmeans(K, N, d, MAX_ITER, vectors.tolist(), centroids.tolist())
