import numpy as np
from matplotlib import pyplot as plt

'''
Utilities file used by main

Contains methods for interacting with files 
'''


def save_data(vectors, clusters, d):
    """
    saves data to a txt file "data.txt"
    each line represents a vector of 2 or 3 dim. the last digit represents the cluster it originally belongs to.

    arguments:
    vectors - list of vectors
    clusters - list of cluster indices - clusters[i] is the index ot the cluster to which vector i belongs
    d - dim of the vectors

    no return value
    """
    # we use the built in numpy method for writing data , for that reason we need the clusters to be in the same ndarray
    data = np.column_stack((vectors, clusters))
    fmt = ("%f", "%f", "%f", "%d") if d == 3 else ("%f", "%f", "%d")
    np.savetxt("data.txt", data, delimiter=", ", fmt=fmt)


def write_to_file(s_clusters, k_clusters, k):
    """
    saves clusters to a txt file "clusters.txt"
    first line is the number of clusters (k) calculated by the Spectral clustering algorithm
    next k lines, each represent a cluster calculated by the Spectral clustering algorithm, vectors are mentioned
        using their index in the data file
    next k lines are the same, but for the K-means algorithm

    arguments:
    s_clusters - list of lists. each list contains the indices of vectors assigned to the cluster by spectral clustering
        eg. s_clusters[0][0] is the index of the first vector in the first cluster
    k_clusters - same for K_means
    k - # of clusters

    no return value
    """
    second_f = open('clusters.txt', 'w+')
    second_f.write(str(k) + "\n")
    for i in range(k):
        second_f.write(str(s_clusters[i])[1: -1] + "\n")
    for i in range(k):
        second_f.write(str(k_clusters[i])[1: -1] + "\n")
    second_f.close()

def save_to_pdf(vectors, spectral, kmeans, dim, k, n, obs_k, sjm, kjm):
    """
    saves a graphic visualisation to a pdf file "clusters.pdf"

    two graphs representing the results of the two algorithms
        different colors represent different clusters

    textual description of algorithms output

    arguments:
    vectors - list of indices representing original assignments to clusters
    spectral - list of indices representing assignments to clusters by spectral clustering
    kmeans - list of indices representing assignments to clusters by spectral clustering
    dim - dim of vectors
    k - original # of clusters generated
    n - # of vectors
    obs_k - # of clusters as calculated by spectral clustering
    sjm - Jaccard measure for spectral clustering
    kjm - Jaccard measure for K_means

    no return value
    """

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



