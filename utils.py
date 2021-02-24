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
        if temp ==0 :
            print(" 0 divvvvvvvvvvv")
        temp = 1/temp
        col = U[:, i]*temp
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
def save_data(vectors, clusters, d):
    data = np.column_stack((vectors[0], clusters[1]))
    fmt = ("%f", "%f", "%f", "%d") if d == 3 else ("%f", "%f", "%d")
    np.savetxt("data.txt", data, delimiter=", ", fmt=fmt)




def QR_iteration_algorithm(A):
    n = A.shape[0]
    Q_bar = np.identity(n)
    for i in range(n):
        Q, R = MGS(A,A.shape[0])
        A = R @ Q
        new_Q_bar = Q_bar @ Q
        ep = (np.absolute(Q_bar) - np.absolute(new_Q_bar)).max()
        if ep < 0.0001:
            return (A,Q_bar)
        Q_bar = new_Q_bar
    return (A,Q_bar)

A = np.array([0, 12, 3, 4, 5, 6, 7, 8, 9],dtype=np.float64).reshape(3, 3)
x = QR_iteration_algorithm(A)
print(x)
