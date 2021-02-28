from invoke import task


@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")


@task(aliases=['del'])
def delete(c):
    c.run("rm *mykmeanssp*.so")


@task(name='run', optional=['Random'], )
def run(c, k=0, n=0, Random=False):
    print("building shared object files")
    c.run("python setup.py build_ext --inplace")  # todo
    c.run("python main.py {n:} {k:} -- {random:}".format(n=n, k=k, random=str(Random)))  # todo
