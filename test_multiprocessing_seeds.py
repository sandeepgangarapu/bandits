import multiprocessing
import random
import numpy as np

def run_simulation(ite):
    local_state = np.random.RandomState(ite)
    print(np.random.normal(0,1,5))

if __name__ == '__main__':
    a_pool = multiprocessing.Pool()
    result = a_pool.map(run_simulation, range(30))