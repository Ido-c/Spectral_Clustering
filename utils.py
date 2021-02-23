import numpy as np
from sklearn.datasets import make_blobs


# Modified Gram-Shmidt
def MGS(A, n):
    U = A.copy()
    R = np.zeros((n, n))
    Q = np.zeros((n, n))
    for i in range(n):
        temp = np.linalg.norm(U[:, i])
        R[i, i] = temp
        col = U[:, i]/temp
        Q[:, i] = col
        for j in range(i+1, n):
            Rij = col @ U[:, j]
            R[i, j] = Rij
            U[:, j] = U[:, j] - Rij*col
    return (Q, R)

# QR iteration

# The Eigengap Heuristic

# Random data generator (from K and N)
def save_data(vectors, clusters, d):
    data = np.column_stack((vectors[0], clusters[1]))
    fmt = ("%f", "%f", "%f", "%d") if d == 3 else ("%f", "%f", "%d")
    np.savetxt("draftinf.txt", data, delimiter=", ", fmt=fmt)

A = np.array([0, 12, 3, 4, 5, 6, 7, 8, 9]).reshape(3, 3)
x = MGS(A,3)
print(x)
    np.savetxt("data.txt", data, delimiter=", ", fmt=fmt)


def QR_iteration_algorithm(A):
    n = A.shape[0]
    Q_bar = np.identity(n)
    for i in range(n):
        Q, R = (0, 0)  # run the gram shmidit algo
        A = R * Q
        new_Q_bar = Q_bar*Q
        ep = abs(Q_bar - new_Q_bar)
        if ep< 0.0001:
            return (A,Q_bar)
        Q_bar = new_Q_bar
    return (A,Q_bar)