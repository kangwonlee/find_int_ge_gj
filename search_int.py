import functools
import itertools
import multiprocessing as mp
import random
import sys

from typing import List, Tuple

import numpy as np
import numpy.linalg as nl


def main(argv):
    '''
    find a matrix of integers
    only integers duing gauss elimination
    and back substitution
    '''

    n_dim = int(argv[1])
    b_mp = False

    if b_mp:
        pool = mp.Pool(mp.cpu_count())
        pool.map(try_ge_over_vec, gen_mat(n_dim))

        pool.close()
        pool.join()
    else:
        for mat in gen_mat(n_dim):
            try_ge_over_vec(mat)


@functools.lru_cache
def len_x(n_dim:int=2):
    '''
    get number of total integers
    '''
    return n_dim*(n_dim+1)


def get_digit(n_dim:int, n_max:int=10) -> Tuple[Tuple[int]]:
    result = []
    for i in range(n_dim):
        one = list(range(-n_max, n_max+1))
        random.shuffle(one)
        result.append(tuple(one))
    return tuple(result)


def gen_mat(n_dim:int=2):
    '''
    generate a list of integers
    '''
    len_x = n_dim * n_dim

    for v in itertools.product(*get_digit(len_x)):
        yield reshape_mat(v, n_dim)


def reshape_mat(v:List[int], n_dim:int=2) -> Tuple[np.ndarray]:
    '''
    reshape a list of integers into a matrix
    '''
    return np.array(v, dtype=float).reshape(n_dim, n_dim)


def reshape_vec(v:List[int], n_dim:int=2) -> Tuple[np.ndarray]:
    '''
    reshape a list of integers into a vector
    '''
    return np.array(v, dtype=float).reshape(n_dim, -1)


def try_ge_over_vec(mat:np.ndarray, epsilon:float=1e-5) -> bool:
    '''
    try gauss elimination over a matrix and a vector
    '''
    n_dim = mat.shape[0]

    if nl.matrix_rank(mat) == n_dim:
        for vv in itertools.product(range(-10, 10), repeat=n_dim):
            vec = reshape_vec(vv, n_dim)
            matAb = np.hstack((mat, vec))
            gauss_elimination_int_only(matAb, epsilon)
            del vec
            del matAb

    del mat


def gauss_elimination_int_only(matAb_org:np.ndarray, epsilon:float=1e-5) -> bool:
    '''
    perform gauss elimination on a matrix and a vector
    see if integer only
    '''

    matAb = matAb_org.copy()

    result = True

    for p in range(matAb.shape[0]):
        for i in range(p+1, matAb.shape[0]):

            if matAb[p, p] == 0:
                result = False
                break

            ratio = - (matAb[i, p] / matAb[p, p])
            matAb[i, p:] += ratio * matAb[p, p:]

            # check if mat[i, p:] integer only
            if not all((matAb[i, p:] % 1) <= epsilon):
                result = False
                # print("not integer only")
                # print(matAb)
                break

        if not result:
            break 

    if result:
        mat = matAb[:, :-1]
        vec = matAb[:, -1]
        x = nl.solve(mat, vec)
        del mat
        del vec
        if not all((x % 1) <= epsilon):
            result = False
            # print("not integer only")
            # print(matAb)
        else:
            print("int only")
            print(matAb_org, x)

    return result


if __name__ == "__main__":
    main(sys.argv)
