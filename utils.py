import numpy as np
from sklearn.datasets import make_blobs


# Modified Gram-Shmidt
def MGS(A, n):
    R = np.zeros((n, n))
    Q = np.zeros((n, n))
    for i in range(n):
        temp = np.linalg.norm(A[:, i])
        R[i, i] = temp
        col = A[:, i]/temp
        Q[:, i] = col
        for j in range(i+1, n):
            R[i, j] = col*A[:, j]


# QR iteration

# The Eigengap Heuristic

# Random data generator (from K and N)
def save_data(vectors,clusters, d):
    data = np.column_stack((vectors[0], clusters[1]))
    fmt = ("%f", "%f", "%f", "%d") if d == 3 else ("%f", "%f", "%d")
    np.savetxt("draftinf.txt", data, delimiter=", ", fmt=fmt)