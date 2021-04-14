import numpy as np
import kmeans_pp
import math


def spectral_clustering(vectors, n):
    """
    Spectral clustering algorithm

    arguments:
    vectors - list of vectors to be clustered
    n - # of vectors

    returns tuple "clusters" containing:
        clusters[0] - list of lists. each list contains the indices of vectors assigned to the cluster
            eg. clusters[0][0][0] is the index of the first vector in the first cluster
        clusters[1] - list of indices mapping each vector to its assigned cluster
            eg. clusters[1][5] is the cluster number to which the 6th vector was assigned
        clusters[2] - ideal # of clusters (k) as calculated using the Eigengap Heuristic
    """
    # Create the weighted adjacency matrix
    WAM = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            temp = find_weight(vectors[i], vectors[j])
            WAM[i, j] = temp
            WAM[j, i] = temp

    # Compute the Diagonal Degree Matrix ^0.5
    DDM = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        DDM[i, i] = np.sum(WAM[i]) ** (-0.5)

    # Compute the normalized graph Laplacian
    Lnorm = np.identity(n) - DDM @ WAM @ DDM

    # Determine k using the Eigengap Heuristic and create list of indices representing the sorted eigenvalues
    eigenvalues, eigenvector_mat = QR_iteration_algorithm(Lnorm)
    order = np.argsort(eigenvalues)
    k = eigengap(eigenvalues)

    # Create a matrix U containing the vectors u_1 : u_k as columns
    # where u_i represents the eigenvector belonging to the ith smallest eigenvalue
    U = eigenvector_mat[:, order[0:k]]

    # Create matrix T  from U by renormalizing each of U's rows to have unit length
    T = np.divide(U.T, np.linalg.norm(U, axis=1)).T
    if np.isnan(T).any():
        raise ZeroDivisionError("division by zero in spectral_clustering")
        # without error nans will only produce warning


    # Treating each row of T as a point in Rk, cluster them into k clusters via the K-means algorithm
    clstr_to_vec, vec_to_clstr = kmeans_pp.k_means_pp(T, k, T.shape[1], T.shape[0], 300)

    return clstr_to_vec, vec_to_clstr, k


def MGS(A):
    """
    Modified Gram-Shmidt algorithm

    arguments:
    A - a matrix of size n*n

    returns a tuple containing a decomposition of A into two matrices, Q and R, where A = QR and:
        [0] Q is orthogonal
        [1] R is upper triangular
    """
    n = A.shape[0]
    U = A.copy()
    R = np.zeros((n, n), dtype=np.float32)
    Q = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        temp = np.linalg.norm(U[:, i])
        if temp == 0:
            raise ZeroDivisionError("division by zero in spectral_clustering")
        R[i, i] += temp
        col = U[:, i] / temp
        Q[:, i] = col
        # col is broadcasted to support matrix multiplication
        # equivalent to computing R[i, j] += col @ U[:, j] for i+1 <= j <= n
        R[i, i + 1:] = col @ U[:, i + 1:]
        # equivalent to computing U[:, j] = U[:, j] - R[i, j] * col for i+1 <= j <= n
        # this is thanks to R being upper triangular
        U -= (R[i, :, np.newaxis] * col).T
    return Q, R


def QR_iteration_algorithm(A):
    """
    QR iteration algorithm - finds eigenvalues and eigenvectors for A

    arguments:
        A - a matrix of size n*n

    returns a tuple (eigenvalues, Q_bar) containing:
        [0] 1D array of the eigenvalues of A
        [1] 2D array of eigenvectors so that Q_bar[i] belongs to the value at eigenvalues[i]
    """
    n = A.shape[0]  # A is (nxn)
    Q_bar = np.identity(n,dtype=np.float32)
    for i in range(n):
        Q, R = MGS(A)
        A = R @ Q
        new_Q_bar = Q_bar @ Q
        ep = (np.absolute(Q_bar) - np.absolute(new_Q_bar)).max()
        if ep <= 0.0001:
            break
        Q_bar = new_Q_bar
        # we dont need the rest of A , so we return only the eigenvalues
    eigenvalues = np.array([A[i, i] for i in range(n)], dtype=np.float32)
    return eigenvalues, Q_bar


def find_weight(x, y):
    """
    find weight

    arguments:
        x, y - vectors

    returns the product of the following function:
        exp{-||x - y||/2) where ||x|| is the Euclidean norm
    """
    dist = np.linalg.norm(x - y)
    return math.exp(-(dist / 2))


# The Eigengap Heuristic
def eigengap(values):
    """
    eigengap - finds the ideal # of clusters (k) for the clustering algorithm

    arguments:
        a list of eigenvalues (floats)

    returns k using the following logic:
        argmax_i(delta_i) where delta_i = abs(values[i] - values[i+1])
    """
    sorted_val = np.sort(values)
    index = 0
    maximum = -1
    for i in range(math.ceil(len(sorted_val) / 2)):
        temp = sorted_val[i + 1] - sorted_val[i]
        if temp > maximum:
            maximum = temp
            index = i
    return index + 1  # +1 because arguments start from zero
