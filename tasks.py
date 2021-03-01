from invoke import task
import time


@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")


@task(aliases=['del'])
def delete(c):
    c.run("rm *mykmeanssp*.pyd")


@task(post=[delete],name='run', optional=['Random'],
      help= {
          "k" : "the number of clusters to be generated",
          "n": "the number of vectors to be generated",
          "Random" : "if not given the the program will ignore the given k and n"})
def run(c, k, n, Random=False):
    """the program generate n random 2d or 3d vectors distributed to k clusters then finds the clusters using both
    spectral clustering and the k-means++ algorithm finally it wright the results in a txt file and return a PDF file
    with the visualisation of the results and the juccrd measure of both algorithms 
    """#todo check with ido
    c.run("python setup.py build_ext --inplace")  # todo write python3.8.5 insted
    c.run("python main.py {n:} {k:} {random:}".format(n=n, k=k, random=("--Random" if Random else "")))  # # todo write python3.8.5 insted


@task(name='find_critical', aliases='find',)
def nk_find(c):
    t1 = t2 = 0
    res = {}
    for k in range(2, 11):
        n = 10
        add = [100, 10, 1]
        for j in range(3):
            while t1 - t2 < ((5 * 60) - 1):
                n +=add[j]
                t1 = time.time()
                print("python main.py {n:} {k:} {random:}".format(n=n, k=k, random="--Random"))
                c.run("python main.py {n:} {k:} {random:}".format(n=n, k=k, random="--Random"))
                t2 = time.time()
            n-=add
    res[k]=n
    file = open('critical .txt', 'w+')
    for key , obj in res.items():
        file.write("k is : {} and n is ".format(key,obj))
    file.close()