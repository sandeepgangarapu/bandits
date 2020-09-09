import multiprocessing as mp
import random
import numpy

def child(n):
    numpy.random.seed(n)  # <-- comment this out to get the fright of your life.
    m = numpy.random.randn(6)
    return m


if __name__ == '__main__':
    N = 20
    pool = mp.Pool()
    results = pool.map(child, range(N))
    for res in results:
        print(res)
