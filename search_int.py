import functools
import itertools

from typing import List, Tuple

import numpy as np
import numpy.linalg as nl


def main():
    '''
    find a matrix of integers
    only integers duing gauss elimination
    and back substitution
    '''

    n_dim = 2

    for matAb in gen_ints(n_dim):
        if gauss_elimination_int_only(matAb):
            print(matAb)


@functools.lru_cache
def len_x(n_dim:int=2):
    '''
    get number of total integers
    '''
    return n_dim*(n_dim+1)


def gen_ints(n_dim:int=2):
    '''
    generate a list of integers
    '''

    for v in itertools.product(range(-10, 10), repeat=(n_dim*n_dim)):
        mat = reshape_mat(v, n_dim)
        if nl.matrix_rank(mat) == n_dim:
            for vv in itertools.product(range(-10, 10), repeat=n_dim):
                vec = reshape_vec(vv, n_dim)
                yield np.hstack((mat, vec))


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


def gauss_elimination_int_only(matAb:np.ndarray, epsilon:float=1e-5) -> bool:
    '''
    perform gauss elimination on a matrix and a vector
    see if integer only
    '''

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
                print("not integer only")
                print(matAb)
                break

        if not result:
            break 

    if result:
        print("int only")
        print(matAb)

    return result


if __name__ == "__main__":
    main()
