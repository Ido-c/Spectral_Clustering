import sklearn.datasets
import numpy as np
a = sklearn.datasets.make_blobs(10, 3, centers=2)
def save_data(vectors,clusters, d):
    data = np.column_stack((vectors[0], clusters[1]))
    fmt = ("%f", "%f", "%f", "%d") if d == 3 else ("%f", "%f", "%d")
    np.savetxt("draftinf.txt", data, delimiter=", ", fmt=fmt)

def usles(x):
    x[2] = 9999999