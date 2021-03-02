from invoke import task
import time


@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")


@task(aliases=['del'])
def delete(c):
    c.run("rm *mykmeanssp*.pyd")


@task(post=[delete], name='run', optional=['Random'],
      help={
          "k": "the number of clusters to be generated",
          "n": "the number of vectors to be generated",
          "Random": "if not given, the program will ignore k and n and take them at random"})
def run(c, k, n, Random=False):
    """the program generate n random 2d or 3d vectors distributed to k clusters then finds the clusters using both
    spectral clustering and the k-means++ algorithm finally it wright the results in a txt file and return a PDF file
    with the visualisation of the results and the juccrd measure of both algorithms 
    """  # todo check with ido
    c.run("python setup.py build_ext --inplace")  # todo write python3.8.5 insted
    c.run("python main.py {n:} {k:} {random:}".format(n=n, k=k, random=("--Random" if Random else "")))  # # todo write python3.8.5 insted


@task(name='find_critical', aliases='find', )
def nk_find(c):
    t1 = t2 = 0
    res = {}
    for k in range(2, 11):
        k = 2
        add = [10, 1]
        for j in range(2):
            while t2 - t1 < ((5 * 60) - 1):
                 += add[j]
                t1 = time.time()
                print("python3.8.5 main.py {n:} {k:} {random:}".format(n=n, k=k, random="--Random"))
                c.run("python3.8.5 main.py {n:} {k:} {random:}".format(n=n, k=k, random="--Random"))
                t2 = time.time()
                print(t2-t1)
            n -= add
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