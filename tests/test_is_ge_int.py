import pathlib
import sys

import pytest


sys.path.insert(
    0,
    str(
        pathlib.Path(__file__).parent.parent.absolute()
    )
)


import is_ge_int as igi


def test_ab2x_list():
    '''
    test ab2x
    '''
    result = igi.ab2x_list(
            '[[ -3.   8.   5.   5.]\n'
            '[ -9.  -3. -10. -10.]\n'
            '[  3.  -8. -10. -10.]]'
        )
    assert result == [
        [-3, 8, 5, 5],
        [-9, -3, -10, -10],
        [3, -8, -10, -10]
    ], result
