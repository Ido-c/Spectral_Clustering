import numpy as np
import utils
import kmeans_pp
from scipy.sparse.csgraph import laplacian


# Form the Weighted Adjacency Matrix W

def spectral_clustering(vectors, n, dim):
    # Create the weighted adjacency matrix
    WAM = np.zeros((n, n), dtype=np.float)
    for i in range(n):
        for j in range(i + 1, n):
            WAM[i, j] = utils.find_weight(vectors[i], vectors[j], dim)
    WAM = WAM + WAM.T

    # Compute the Diagonal Degree Matrix ^0.5
    DDM = np.zeros((n, n), dtype=np.float)
    for i in range(n):
        DDM[i, i] = np.sum(WAM[i]) ** (-0.5)

    # Compute the normalized graph Laplacian

    Lnorm = np.identity(n) - DDM @ WAM @ DDM

    # Determine k and obtain the first k eigenvectors of Lnorm
    eigenvalues, eigenvector_mat = utils.QR_iteration_algorithm(Lnorm)
    order = np.argsort(eigenvalues)
    k = utils.eigengap(eigenvalues)
    print("k is ", k)  # todo
    # Let U be the matrix containing the vectors u1; : : : ; uk as columns
    U = eigenvector_mat[:, order[0:k]]
    # Form the matrix T  from U by renormalizing each of U's rows to have unit length
    T = np.divide(U.T, np.linalg.norm(U, axis=1)).T

    # Treating each row of T as a point in Rk, cluster them into k clusters via the K-means algorithm
    clstr_to_vec, vec_to_clstr = kmeans_pp.k_means_pp(T, k, T.shape[1], T.shape[0], 300)

    # Assign the original point xi to cluster j if and only if row i of the matrix T was assigned to cluster j]
    return clstr_to_vec, vec_to_clstr, k
