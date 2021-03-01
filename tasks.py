from invoke import task


@task
def build(c):
    c.run("python setup.py build_ext --inplace")


@task(aliases=['del'])
def delete(c):
    c.run("rm *mykmeanssp*.so")


@task(name='run', optional=['Random'], )
def run(c, k, n, Random=False):
    print("building shared object files")
    c.run("python setup.py build_ext --inplace")  # todo
    c.run("python main.py {n:} {k:} -- {random:}".format(n=n, k=k, random=str(Random)))  # todo
