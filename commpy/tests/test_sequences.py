# Authors: CommPy contributors
# License: BSD 3-Clause

from numpy import array
from numpy.random import seed
from numpy.testing import run_module_suite, assert_raises, assert_equal

from commpy.sequences import pnsequence


def test_pnsequence():
    # Test the raises of errors
    with assert_raises(ValueError):
        pnsequence(4, '001', '1101', 2**4 - 1)
        pnsequence(4, '0011', '110', 2 ** 4 - 1)

    # Test output with
    assert_equal(pnsequence(4, '0011', '1101', 7), array((1, 1, 0, 0, 1, 0, 1), int),
                 err_msg='Pseudo-noise sequence is not the one expected.')

if __name__ == "__main__":
    seed(17121996)
    run_module_suite()
