import numpy as np
import math
import utils
# Form the Weighted Adjacency Matrix W

def spectral_clustering(vectors, n, dim):
    # Create the weighted adjacency matrix
    WAM = np.zeros((n, n), dtype=np.float)
    for i in range(1, n):
        for j in range(i + 1, n):
            WAM[i, j] = find_weight(vectors[i], vectors[j], dim)
    WAM = WAM + WAM.T

    # Compute the Diagonal Degree Matrix ^0.5
    DDM = np.zeros((n, n), dtype=np.float)
    for i in range(n):
        DDM[i, i] = np.sum(WAM[i])**(-0.5)

    # Compute the normalized graph Laplacian
    Lnorm = np.identity(n) - DDM @ WAM @ DDM

    # Determine k and obtain the first k eigenvectors of Lnorm
    eigenvalues, eigenvector_mat = utils.QR_iteration_algorithm(Lnorm)
    order = np.argsort(eigenvalues)
    k = utils.eigengap(eigenvalues)

    # Let U be the matrix containing the vectors u1; : : : ; uk as columns
    U = eigenvector_mat[order[0:k+1]]

    # Form the matrix T  from U by renormalizing each of U's rows to have unit length

# Treating each row of T as a point in Rk, cluster them into k clusters via the K-means algorithm

# Assign the original point xi to cluster j if and only if row i of the matrix T was assigned to cluster j


def find_weight(x, y, dim):
    dist = np.linalg.norm(x - y)
    return math.exp(-(dist/2))

