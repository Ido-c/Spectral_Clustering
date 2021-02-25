import sklearn.datasets
import numpy as np
from matplotlib import pyplot as plt
a = sklearn.datasets.make_blobs(10, 3, centers=2)


def save_data(vectors, clusters, d):
    data = np.column_stack((vectors[0], clusters[1]))
    fmt = ("%f", "%f", "%f", "%d") if d == 3 else ("%f", "%f", "%d")
    np.savetxt("draftinf.txt", data, delimiter=", ", fmt=fmt)


def usles(x):
    x[2] = 9999999

# A = np.array([0, 12, 3, 4, 5, 6, 7, 8, 9],dtype=np.float64).reshape(3, 3)
# x = QR_iteration_algorithm(A)
# print(x)
fig = plt.figure()
plot1 = fig.add_subplot(122)
plot1.plot(range(10))
plot1.text(.1,.1,"some text")
plot2 = fig.add_subplot(121)
plot2.plot(a[1])

fig.savefig("testing.pdf")

print("done")