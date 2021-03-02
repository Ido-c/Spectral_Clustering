import mykmeanssp as kmns
import numpy as np

'''
K-means clustering algorithm

arguments:
vectors - list of vectors to be clustered
K - desired # of clusters
d - dim of vectors
N - # of vectors
MAX-ITER - maximum number of iterations for the algorithm

returns tuple "clusters" containing:
    clusters[0] - list of lists. each list contains the indices of vectors assigned to the cluster
        eg. clusters[0][0][0] is the index of the first vector in the first cluster
    clusters[1] - list of indices mapping each vector to its assigned cluster
        eg. clusters[1][5] is the cluster number to which the 6th vector was assigned
'''
def k_means_pp(vectors, K, d, N, MAX_ITER):

    np.random.seed(0)  # pick the seed val to be 0
    # indices = np.arange(N)
    centroids_count = 1

    centroid_indices = np.zeros(K, dtype=np.int16)
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
    vectors = vectors.reshape(vectors.size)
    centroids = centroids.reshape(centroids.size)
    clusters = kmns.kmeans(K, N, d, MAX_ITER, list(vectors), list(centroids))  # todo
    return clusters
