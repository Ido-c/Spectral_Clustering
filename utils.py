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
def Eigengap(values):
    sorted = np.sort(values)
    index = 0
    max = -1
    for i in range(len(sorted)//2):
        temp = abs(sorted[i]-sorted[i + 1])
        if temp > max:
            max = temp
            index = i
    return index


# Random data generator (from K and N)
def save_data(vectors,clusters, d):
    data = np.column_stack((vectors[0], clusters[1]))
    fmt = ("%f", "%f", "%f", "%d") if d == 3 else ("%f", "%f", "%d")
    np.savetxt("draftinf.txt", data, delimiter=", ", fmt=fmt)

A = np.array([0, 12, 3, 4, 5, 6, 7, 8, 9]).reshape(3, 3)
x = MGS(A,3)
print(x)