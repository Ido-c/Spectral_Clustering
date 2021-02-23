import numpy as np
from sklearn.datasets import make_blobs


# Modified Gram-Shmidt

# QR iteration

# The Eigengap Heuristic

# Random data generator (from K and N)
def save_data(vectors, clusters, d):
    data = np.column_stack((vectors[0], clusters[1]))
    fmt = ("%f", "%f", "%f", "%d") if d == 3 else ("%f", "%f", "%d")
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