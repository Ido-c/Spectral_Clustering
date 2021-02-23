import numpy as np
from sklearn.datasets import make_blobs
import utils
'''
maxcap
'''
MAX_CAP_N = 200
MAX_CAP_K = 10


def main(n, k, random):
    if random: # choose at random from range [max_capacity/2, max_capacity]
        k = np.random.randint(MAX_CAP_K//2, MAX_CAP_K + 1)
        n = np.random.randint(MAX_CAP_N//2, MAX_CAP_N + 1)
    d=np.random.choice(2,3)

    #   arguments assertion
    flag = False
    if k > n:
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
    vectors,clusters = make_blobs(n,d,centers=k)

    # Create 1st txt file
    utils.save_data(vectors,clusters,d)
    # Create 2nd txt file and put char for K

    # Run Kmeanspp and put clusters in 2nd file

    # Run Spectral Clustering and put clusters in 2nd file

    # Create pdf file

    # Compute Jaccard measure for both algo's and put in pdf

    # if __name__ == "__main__":
    #     main()