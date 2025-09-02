import multiprocessing as mp
import numpy as np
from random import gauss
from itertools import product

def run_combination(comb_, seed, nIter, maxHP, phi, coeffs):
    output2 = []
    for iter_ in range(int(nIter)):
        p = seed
        hp = 0
        while True:
            p = (1 - phi) * coeffs['forecast'] + phi * p + coeffs['sigma'] * gauss(0, 1)
            cP = p - seed
            hp += 1
            if cP > comb_[0] or cP < -comb_[1] or hp > maxHP:
                output2.append(cP)
                break

    mean, std = np.mean(output2), np.std(output2)
    return comb_[0], comb_[1], mean, std, mean / std


def _run_combination_wrapper(args):
    return run_combination(*args)


def run_batch(coeffs, rPT, rSLm, nIter=1e5, maxHP=100, seed=0, n_jobs=4):
    phi = 2 ** (-1. / coeffs['hl'])
    combinations = list(product(rPT, rSLm))

    args = [(comb_, seed, nIter, maxHP, phi, coeffs) for comb_ in combinations]

    with mp.Pool(processes=n_jobs) as pool:
        results = list(pool.imap(_run_combination_wrapper, args))

    return results
