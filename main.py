import numpy as np
from sklearn.datasets import make_blobs
MAX_CAP_N = 200
MAX_CAP_K = 10


def main(n, k, random):
    if random:
        k = np.random.randint(MAX_CAP_K//2, MAX_CAP_K + 1)
        n = np.random.randint(MAX_CAP_N//2, MAX_CAP_N + 1)
    d=np.random.choice(2,3)




    # Parse commandline arguments

    # assert arguments

    # Print max capacity

    # Generate random data with indexed data points
    vectors = make_blobs(n,d,centers=k)
    # Create 1st txt file
    data = open("data.txt", "w+")
    data.write("")
    # Create 2nd txt file and put char for K

    # Run Kmeanspp and put clusters in 2nd file

    # Run Spectral Clustering and put clusters in 2nd file

    # Create pdf file

    # Compute Jaccard measure for both algo's and put in pdf

    # if __name__ == "__main__":
    #     main()