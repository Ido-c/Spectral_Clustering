import numpy as np
import kmeans_pp
import math


'''
Spectral clustering algorithm

arguments:
vectors - list of vectors to be clustered
n - # of vectors
dim - dim of vectors

returns tuple "clusters" containing:
    clusters[0] - list of lists. each list contains the indices of vectors assigned to the cluster
        eg. clusters[0][0][0] is the index of the first vector in the first cluster
    clusters[1] - list of indices mapping each vector to its assigned cluster
        eg. clusters[1][5] is the cluster number to which the 6th vector was assigned
    clusters[2] - ideal # of clusters (k) as calculated using the Eigengap Heuristic 
'''

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

    # Determine k using the Eigengap Heuristic and create list of indices representing the sorted eigenvalues
    eigenvalues, eigenvector_mat = QR_iteration_algorithm(Lnorm)
    order = np.argsort(eigenvalues)
    k = eigengap(eigenvalues)

    # Create a matrix U containing the vectors u_1 : u_k as columns
    # where u_i represents the eigenvector belonging to the ith smallest eigenvalue
    U = eigenvector_mat[:, order[0:k]]

    # Create matrix T  from U by renormalizing each of U's rows to have unit length
    T = np.divide(U.T, np.linalg.norm(U, axis=1)).T

    # Treating each row of T as a point in Rk, cluster them into k clusters via the K-means algorithm
    clstr_to_vec, vec_to_clstr = kmeans_pp.k_means_pp(T, k, T.shape[1], T.shape[0], 300)

    return clstr_to_vec, vec_to_clstr, k

'''
Modified Gram-Shmidt Algorithm

arguments:
A - a matrix of size n*n

returns a tuple containing a decomposition of A into two matrices, Q and R, where A = QR and:
    1. Q is orthogonal matrix Q 
    2. R is upper triangular
'''
def MGS(A):
    n = A.shape[0]
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
        Q, R = MGS(A)
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