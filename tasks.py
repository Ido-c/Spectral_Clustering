from invoke import task
import time


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


@task(name='find_critical', aliases='find')
def nk_find(c):
    res = {}
    for k in range(2, 11):
        n=270
        add = [10, 1]
        for j in range(2):
            t1 = t2 = 0
            while t2 - t1 < ((5 * 60) - 1):
                n+= add[j]
                t1 = time.time()
                print("python3.8.5 main.py {n:} {k:} {random:}".format(n=n, k=k, random="--no-Random"))
                c.run("python3.8.5 main.py {n:} {k:} {random:}".format(n=n, k=k, random="--no-Random"))
                t2 = time.time()
                print(t2-t1)
            n -= add[j]
        print("finished while loop")
    res[k] = n
    file = open('critical.txt', 'w+')
    for key, obj in res.items():
        file.write(f"k is : {key} and n is {obj}")
    file.close()

@task()
def find2(c):
    n=300
    t1 =0
    t2=0
    k=1
    while t2 -t1<300:
        k+=1
        t1 = time.time()
        print("python3.8.5 main.py {n:} {k:} {random:}".format(n=n, k=k, random="--Random"))
        c.run("python3.8.5 main.py {n:} {k:} {random:}".format(n=n, k=k, random="--Random"))
        t2 = time.time()
        print(t2-t1)
    file = open('criticalw.txt', 'w+')