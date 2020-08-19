# Authors: CommPy contributors
# License: BSD 3-Clause

from numpy import array
from numpy.testing import run_module_suite, assert_array_equal

from commpy.utilities import dec2bitarray


def test_dec2bitarray():
    # Assert result
    assert_array_equal(dec2bitarray(17, 8), array((0, 0, 0, 1, 0, 0, 0, 1)))
    assert_array_equal(dec2bitarray((17, 12), 5), array((1, 0, 0, 0, 1, 0, 1, 1, 0, 0)))


if __name__ == "__main__":
    run_module_suite()
