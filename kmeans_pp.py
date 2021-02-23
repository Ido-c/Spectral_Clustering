import mykmeanssp as kmns

import numpy as np
import argparse

def k_means_pp(vectors, K, d, N):
    """variables are:
    centroids_count : count hpe many centroids have been calculated
    distance : the distence from each vector to the closest centroid
    centroid_indices : ndarry containing the indices of the vectors chosen to be centroids
    centroids : ndarray containing the vectors chosen to be centroids
    """
    np.random.seed(0)  # pick the seed val to be 0
    # indices = np.arange(N)
    centroids_count = 1

    centroid_indices = np.zeros(K, dtype=np.int)
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
    s = ""
    for elm in centroid_indices:
        s = s + str(elm) + ", "
    print(s[:-2])
    return centroids


parser = argparse.ArgumentParser()
parser.add_argument('K', type=int, help="# of clusters")
parser.add_argument('N', type=int, help="# of vectors")
parser.add_argument('d', type=int, help="# of elements in vector")
parser.add_argument('MAX_ITER', type=int, help="max # of iterations")
parser.add_argument('filename', type=str, help="path of vile containing vectors")
args = parser.parse_args()

df =
fcentroids = k_means_pp(fvectors, args.K, args.d, args.N)
fvectors = list(fvectors.reshape(fvectors.size))
fcentroids = list(fcentroids.reshape(fcentroids.size))
kmns.kmeans(args.K, args.N, args.d, args.MAX_ITER, fvectors, fcentroids)
