import ast
import sys

from typing import List

import numpy as np

import search_int as si


def main(argv:List[str]):
    '''
    command line arguments is a list of integers
    is the given matrix of integers GE int ?

    % python is_ge_int.py "[[ -3.   8.   5.   5.]
 [ -9.  -3. -10. -10.]
 [  3.  -8. -10. -10.]]"
    '''
    matAB = np.array(ab2x_list(argv[1]), dtype=float)

    si.gauss_elimination_int_only(matAB)


def ab2x_list(ab:str) -> List[int]:
    '''
    convert a string to a list of list of integers
    '''
    return ast.literal_eval(
        ', '.join(
            map(
                lambda x: x,
                ab.replace('.', ',')
                    .splitlines()
            )
        )
    )


if __name__ == "__main__":
    main(sys.argv)
