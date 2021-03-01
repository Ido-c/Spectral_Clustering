import numpy as np
from matplotlib import pyplot as plt


def save_data(vectors, clusters, d):
    # we use the built in numpy method for writing data , for that reason we need the clusters to be in the same ndarray
    data = np.column_stack((vectors, clusters))
    fmt = ("%f", "%f", "%f", "%d") if d == 3 else ("%f", "%f", "%d")
    np.savetxt("data.txt", data, delimiter=", ", fmt=fmt)


def write_to_file(file, clusters, k):
    for i in range(k):
        file.write(str(clusters[i])[1: -1] + "\n")


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
    plot1.title.set_text('Normalized Spectral Clustering')
    plot2.title.set_text('K-means')
    plot3 = fig.add_subplot(2, 2, 3)
    plot3.set_axis_off()
    text = f"""Data was generated from the values:
    n = {n:}, k = {k:}
    The k that was used for both algorithms was {obs_k:}
    The Jaccard measure for Spectral clustering: {sjm:}
    The Jaccard measure for K-means: {kjm:}
    """
    plot3.text(1.1, 0.4, text, ha="center")
    fig.savefig("clusters.pdf")



