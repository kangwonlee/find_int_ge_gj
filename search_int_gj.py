import functools
import itertools
import multiprocessing as mp
import random

from typing import List, Tuple

import numpy as np
import numpy.linalg as nl


def main():
    '''
    find a matrix of integers
    only integers duing gauss jordan
    and back substitution
    '''

    n_dim = 2
    b_mp = False

    if b_mp:
        pool = mp.Pool(mp.cpu_count())
        pool.map(try_gj_over_vec, gen_mat(n_dim))

        pool.close()
        pool.join()
    else:
        for mat in gen_mat(n_dim):
            try_gj_over_vec(mat)


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


def try_gj_over_vec(mat:np.ndarray, epsilon:float=1e-5) -> bool:
    '''
    try gauss jordan over a matrix and a vector
    '''
    n_dim = mat.shape[0]

    if nl.matrix_rank(mat) == n_dim:
        gauss_jordan_int_only(mat, epsilon)

    del mat


def gauss_jordan_int_only(matA:np.ndarray, epsilon:float=1e-5) -> bool:
    '''
    perform gauss jordan on a matrix and a vector
    see if integer only
    '''

    matAI = np.hstack((matA, np.eye(matA.shape[0])))

    result = True

    for p in range(matAI.shape[0]):
        for i in range(0, matAI.shape[0]):
            if i == p:
                continue

            if matAI[p, p] == 0:
                result = False
                break

            ratio = - (matAI[i, p] / matAI[p, p])
            matAI[i, :] += ratio * matAI[p, :]

            # check if mat[i, :] integer only
            if not all((matAI[i, :] % 1) <= epsilon):
                result = False
                # print("not integer only")
                # print(matAb)
                break

        if not result:
            break 

    if result:
        matInv = matAI[:, matA.shape[0]:]

        if all(
            np.isclose(
                matA @ matInv,
                np.eye(matA.shape[0])
            ).flatten().all()
        ):
            print("int only")
            print(matAI)

    return result


if __name__ == "__main__":
    main()
