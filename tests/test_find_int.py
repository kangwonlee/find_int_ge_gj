import os
import pathlib
import sys


import numpy as np
import numpy.testing as nt
import pytest


sys.path.insert(
    0,
    str(pathlib.Path(__file__).parent.parent.absolute())
) 


import find_int as fi


@pytest.fixture
def ndim() -> int:
    return 4


@pytest.fixture
def len_x(ndim:int) -> int:
    return ndim * (ndim + 1)


@pytest.fixture
def bounds(len_x:int):
    return [(-100, 100)] * len_x


@pytest.fixture
def x() -> np.ndarray:
    return np.array((
        4, 3, 2, 1,
        3, 4, 2, 1,
        2, 3, 4, 1,
        1, 2, 3, 4,
        1, 2, 3, 4
    ))

@pytest.fixture
def mat_a_b() -> np.ndarray:
    return np.array(
        (
            (4, 3, 2, 1, 1),
            (3, 4, 2, 1, 2),
            (2, 3, 4, 1, 3),
            (1, 2, 3, 4, 4),
        )
    )


def test_reshape(x, ndim, mat_a_b):
    result = fi.reshape(x, ndim)

    nt.assert_equal(result, mat_a_b)


def test_eval_ge(x):
    result = fi.eval_ge(x)
    assert result >= -2000
