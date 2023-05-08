"""
Find integer matrix producing only integers 
during Gauss Eliminations or Gauss Jordan Eliminations
"""

import ast
import pathlib

import numpy as np
import numpy.linalg as nl
import scipy.optimize as so


def main():
    n_dim = 4
    len_x = n_dim * (n_dim + 1)

    bounds = [(0, 100)] * len_x

    npop = 1000

    cache = pathlib.Path('cache.txt')

    def show_best(xk, convergence):
        print(convergence)
        print(reshape(xk, n_dim))
        print(eval_ge(xk, n_dim))

        with cache.open('w') as f:
            f.write(str(xk.tolist()))

        return False

    if cache.exists():
        with cache.open('r') as f:
            txt = f.read()
        x0 = ast.literal_eval(txt)
    else:
        x0 = None

    result = so.differential_evolution(
        eval_ge, bounds, args=(n_dim,), popsize=npop,
        workers=-1, updating='deferred',
        callback=show_best,
        x0=x0,
    )
    print(result)
    print(reshape(result.x, n_dim))


def reshape(x:np.ndarray, n_dim:int=4) -> np.ndarray:
    a_len = n_dim * n_dim
    b_end = a_len + n_dim
    a_vec = x[:a_len]

    matA = np.reshape(a_vec, (n_dim, n_dim))
    b_vec = np.reshape(x[a_len:b_end], (n_dim, -1))

    matAB = np.hstack((matA, b_vec))

    return np.array(matAB, dtype=float)


def eval_ge(x:np.ndarray, n_dim:int=4) -> float:
    """
    Evaluate gene for Gauss Elimination
    """
    result = sum(abs(x) - abs(x).astype(int))

    matAB = reshape(x, n_dim)

    if nl.matrix_rank(matAB) < n_dim:
        result += 10.0
    else:
        for p in range(n_dim):
            for i in range(p+1, n_dim):
                result += ge_step(matAB, p, i)

    return result


def ge_step(matAb:np.ndarray, p:int, i:int, epsilon=1e-3) -> float:
    """
    Row operation between the pivot row and ith row
    """
    assert isinstance(i, int)
    assert isinstance(p, int)
    assert i > p, f"i={i} <= p={p}"

    result = 0.0

    if abs(matAb[i, p]) < epsilon:
        result += 1.0
    else:
        ratio = -(matAb[i, p] / matAb[p, p])
        matAb[i, p] = 0
        matAb[i, p+1:] += ratio * matAb[p, p+1:]

        for aij in matAb[i, p+1:]:
            result += (abs(aij) - int(abs(aij)))

    return result


if "__main__" == __name__:
    main()
