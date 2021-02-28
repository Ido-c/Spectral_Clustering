import math
import numpy as np
import errors
from matplotlib import pyplot as plt


def find_weight(x, y, dim):
    dist = np.linalg.norm(x - y)
    return math.exp(-(dist / 2))


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


# The Eigengap Heuristic
def eigengap(values):
    sorted = np.sort(values)
    index = 0
    max = -1
    for i in range(len(sorted) // 2):
        temp = abs(sorted[i] - sorted[i + 1])
        if temp > max:
            max = temp
            index = i
    return index



def save_data(vectors, clusters, d):
    data = np.column_stack((vectors, clusters))
    fmt = ("%f", "%f", "%f", "%d") if d == 3 else ("%f", "%f", "%d")
    np.savetxt("data.txt", data, delimiter=", ", fmt=fmt)


def write_to_file(file, clusters, k, n):
    for i in range(k):
        lst = [clusters[i * (n + 1) + j] for j in range(1, clusters[i * (n + 1)] + 1)]
        file.write(str(lst)[1: -1] + "\n")


def jaccard_measure(org, x):

    return


def save_to_pdf(vectors, spectral, kmeans, dim, k, n, obs_k, sjm, kjm):
    fig = plt.figure()
    if dim == 3:
        plot1 = fig.add_subplot(221, projection='3d')
        plot2 = fig.add_subplot(222, projection='3d')
        plot1.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], c=spectral)
        plot2.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], c=kmeans)
    else:
        plot1 = fig.add_subplot(221)
        plot2 = fig.add_subplot(222)
        plot1.scatter(vectors[:, 0], vectors[:, 1], c=spectral)
        plot2.scatter(vectors[:, 0], vectors[:, 1], c=kmeans)
    plot3 = fig.add_subplot(2, 2, 3)
    plot3.set_axis_off()
    text = "Data was generated from the values:\nn = " + str(n) \
           + " , k = " + str(k) + "\nThe k that was used for both algorithms was " + str(obs_k) \
           + "\nThe Jaccard measure for Spectral clustering: " + str(sjm) \
           + "\nThe Jaccard measure for K-means: " + str(kjm)
    plot3.text(1.1, 0.6, text, ha="center")
    fig.savefig("clusters.pdf")


# QR iteration
def QR_iteration_algorithm(A):
    n = A.shape[0]
    Q_bar = np.identity(n)
    for i in range(n):
        Q, R = MGS(A, A.shape[0])
        A = R @ Q
        new_Q_bar = Q_bar @ Q
        ep = (np.absolute(Q_bar) - np.absolute(new_Q_bar)).max()
        if ep < 0.0001:
            break
        Q_bar = new_Q_bar
    eigenvalues = np.array([A[i, i] for i in range(n)], dtype=np.float64)
    return eigenvalues, Q_bar
