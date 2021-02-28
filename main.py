import argparse

import numpy as np
from sklearn.datasets import make_blobs
import utils
import spectral_clustering
import kmeans_pp

'''
maxcap
'''
MAX_CAP_N = 200
MAX_CAP_K = 10


def main(n, k, random):
    if random:  # choose at random from range [max_capacity/2, max_capacity]
        k = np.random.randint(MAX_CAP_K // 2, MAX_CAP_K + 1)
        n = np.random.randint(MAX_CAP_N // 2, MAX_CAP_N + 1)
    d = np.random.choice((2, 3))

    #   arguments assertion
    flag = False
    if k >= n:
        print("# of clusters must be smaller than # of points")
        flag = True
    if k <= 0:
        print("# of clusters must be a positive non-zero number")
        flag = True
    if n <= 0:
        print("# of points must be a positive non-zero number")
        flag = True
    if flag:
        return

    # Print max capacity
    # Generate random data with indexed data points
    vectors, clusters = make_blobs(n_samples=n, n_features=d, centers=k)
    print(len(clusters[clusters == 0]))
    vectors,clusters = make_blobs(n_samples=n,n_features=d,centers=k)
    x, k = spectral_clustering.spectral_clustering(vectors, n, d) #todo erase x, k
    print("k is",k)
    print("now ony kmeans") #todo
    kmeans_pp.k_means_pp(vectors, k, d, n, 300)
    # Create 1st txt file
    utils.save_data(vectors, clusters, d)

    # Run Spectral Clustering and put clusters in 2nd file
    x, obs_k = spectral_clustering.spectral_clustering(vectors, n, d)  # todo erase x, k
    spectral_clusters = utils.create_cluster_vector(x, n, obs_k)

    # Create 2nd txt file and put char for K
    second_f = open('clusters.txt', 'w+')
    second_f.write(str(obs_k) + "\n")
    utils.write_to_file(second_f, x, obs_k, n)

    # Run Kmeanspp and put clusters in 2nd file
    print("now ony kmeans")  # todo
    x = kmeans_pp.k_means_pp(vectors, obs_k, d, n, 300)
    kmeans_clusters = utils.create_cluster_vector(x, n, obs_k)
    utils.write_to_file(second_f, x, obs_k, n)
    second_f.close()

    # Calculate Jaccard measure
    sjm = utils.jaccard_measure(clusters, spectral_clusters)
    kjm = utils.jaccard_measure(clusters, kmeans_clusters)

    # Create pdf file
    utils.save_to_pdf(vectors, spectral_clusters, kmeans_clusters, d, k, n, obs_k, sjm, kjm)


# Compute Jaccard measure for both algo's and put in pdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, help="# of vectors")
    parser.add_argument('k', type=int, help="# of clusters")
    parser.add_argument('--Random',  help="random k and n",default=True,action='store_false')
    args = parser.parse_args()
    #random = (args.random == "True") # todo check for a better solution
    # k = args.k
    # n = args.n

    print("n is:  ", args.n)
    print("k is:  ", args.k)
    print("random is:  ", bool(args.Random))
    #main(args.n, args.k, bol(args.random))
