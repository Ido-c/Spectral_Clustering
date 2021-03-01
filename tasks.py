from invoke import task


@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")


@task(aliases=['del'])
def delete(c):
    c.run("rm *mykmeanssp*.so")


@task(name='run', optional=['Random'], )
def run(c, k, n, Random=False):
    print("building shared object files")
    c.run("python setup.py build_ext --inplace")  # todo
    print("python main.py {n:} {k:} {random:}".format(n=n, k=k, random=("--Random" if Random else "")))  # todo
    c.run("python main.py {n:} {k:} {random:}".format(n=n, k=k, random=("--Random" if Random else "")))  # todo
