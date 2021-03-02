import argparse
import numpy as np
from sklearn.datasets import make_blobs
import utils
import spectral_clustering
import kmeans_pp
import mykmeanssp as kmns

'''
maxcap
'''
MAX_CAP_N = 200
MAX_CAP_K = 10


def main(n, k, random):
    if random:  # values for k & n are chosen at random from range [max_capacity/2, max_capacity]
        k = np.random.randint(MAX_CAP_K // 2, MAX_CAP_K + 1)
        n = np.random.randint(MAX_CAP_N // 2, MAX_CAP_N + 1)
    d = np.random.choice((2, 3))

    # checking the arguments arguments

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
    print(f"max capacity of k :{MAX_CAP_K}\n"
          f"max capacity of n : {MAX_CAP_N}")

    # Generate random data with indexed data points
    vectors, clusters = make_blobs(n_samples=n, n_features=d, centers=k)
    # Create 1st txt file
    utils.save_data(vectors, clusters, d)

    # Run Spectral Clustering and put clusters in 2nd file
    s_clstr_to_vec, s_vec_to_clstr, obs_k = spectral_clustering.spectral_clustering(vectors, n, d)  # todo erase x, k

    # Create 2nd txt file and write K in it
    second_f = open('clusters.txt', 'w+')
    second_f.write(str(obs_k) + "\n")
    utils.write_to_file(second_f, s_clstr_to_vec, obs_k)

    # Run Kmeanspp and put clusters in 2nd file
    k_clstr_to_vec, k_vec_to_clstr = kmeans_pp.k_means_pp(vectors, obs_k, d, n, 300)
    utils.write_to_file(second_f, k_clstr_to_vec, obs_k)
    second_f.close()

    # Calculate Jaccard measure
    temp = clusters.tolist()
    sjm = kmns.jaccard(temp, s_vec_to_clstr)
    kjm = kmns.jaccard(temp, k_vec_to_clstr)

    # Create pdf file
    utils.save_to_pdf(vectors, s_vec_to_clstr, k_vec_to_clstr, d, k, n, obs_k, sjm, kjm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, help="# of vectors")
    parser.add_argument('k', type=int, help="# of clusters")
    parser.add_argument('--Random', help="random k and n", default=True, action='store_false')
    args = parser.parse_args()

    main(args.n, args.k, args.Random)
