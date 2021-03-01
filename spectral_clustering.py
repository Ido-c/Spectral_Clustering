import numpy as np
import kmeans_pp
import math

# Form the Weighted Adjacency Matrix W
def spectral_clustering(vectors, n, dim):

    # Create the weighted adjacency matrix
    WAM = np.zeros((n, n), dtype=np.float)
    for i in range(n):
        for j in range(i + 1, n):
            WAM[i, j] = find_weight(vectors[i], vectors[j], dim)
    WAM = WAM + WAM.T

    # Compute the Diagonal Degree Matrix ^0.5
    DDM = np.zeros((n, n), dtype=np.float)
    for i in range(n):
        DDM[i, i] = np.sum(WAM[i]) ** (-0.5)

    # Compute the normalized graph Laplacian
    Lnorm = np.identity(n) - DDM @ WAM @ DDM

    # Determine k and obtain the first k eigenvectors of Lnorm
    eigenvalues, eigenvector_mat = QR_iteration_algorithm(Lnorm)
    order = np.argsort(eigenvalues)
    k = eigengap(eigenvalues)

    # Let U be the matrix containing the vectors u1; : : : ; uk as columns
    U = eigenvector_mat[:, order[0:k]]

    # Form the matrix T  from U by renormalizing each of U's rows to have unit length
    T = np.divide(U.T, np.linalg.norm(U, axis=1)).T

    # Treating each row of T as a point in Rk, cluster them into k clusters via the K-means algorithm
    clstr_to_vec, vec_to_clstr = kmeans_pp.k_means_pp(T, k, T.shape[1], T.shape[0], 300)

    # Assign the original point xi to cluster j if and only if row i of the matrix T was assigned to cluster j]
    return clstr_to_vec, vec_to_clstr, k


# Modified Gram-Shmidt
def MGS(A, n):
    U = A.copy()
    R = np.zeros((n, n))
    Q = np.zeros((n, n))
    for i in range(n):
        temp = np.linalg.norm(U[:, i])
        R[i, i] = temp
        if temp == 0:
            errors.division_by_zero()  # todo errors
            return
        col = U[:, i] / temp
        Q[:, i] = col
        for j in range(i + 1, n):
            Rij = col @ U[:, j]
            R[i, j] = Rij
            U[:, j] = U[:, j] - Rij * col
    return Q, R


# QR iteration
def QR_iteration_algorithm(A):
    n = A.shape[0]  # A is (nxn)
    Q_bar = np.identity(n)
    for i in range(n):
        Q, R = MGS(A, A.shape[0])
        A = R @ Q
        new_Q_bar = Q_bar @ Q
        ep = (np.absolute(Q_bar) - np.absolute(new_Q_bar)).max()
        if ep < 0.0001:
            break
        Q_bar = new_Q_bar
        # we dont need the rest of A , so we return only the eigenvalues
    eigenvalues = np.array([A[i, i] for i in range(n)], dtype=np.float64)
    return eigenvalues, Q_bar


def find_weight(x, y, dim):
    dist = np.linalg.norm(x - y)
    return math.exp(-(dist / 2))


# The Eigengap Heuristic
def eigengap(values):
    sorted = np.sort(values)
    index = 0
    max = -1
    for i in range(math.ceil(len(sorted) / 2)):
        temp = abs(sorted[i] - sorted[i + 1])
        if temp > max:
            max = temp
            index = i
    return index + 1