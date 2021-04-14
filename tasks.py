from invoke import task


@task(name='run',
      help={
          "k": "the number of clusters to be generated",
          "n": "the number of vectors to be generated",
          "Random": "if not specified, the program will ignore k and n and choose them at random"})
def run(c, k, n, Random=True):
    """
    Runs a program which computes ond compares Kmeans clustering vs. Spectral clustering for given variables.
    The program generates n random 2d or 3d vectors distributed to k clusters.
    next the program finds the clusters using both spectral clustering and the k-means++ algorithm.
    The results of the test are saved to a txt file and a visualization of the results and the comparison are
    saved in a PDF
    """
    c.run("python3.8.5 setup.py build_ext --inplace")
    c.run("python3.8.5 main.py {n:} {k:} {random:}".format(n=n, k=k, random=("--Random" if Random else "")))