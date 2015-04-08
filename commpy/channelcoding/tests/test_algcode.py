
# Authors: Veeresh Taranalli <veeresht@gmail.com>
# License: BSD 3-Clause

from numpy import array
from numpy.testing import assert_array_equal
from commpy.channelcoding.algcode import cyclic_code_genpoly

class TestAlgebraicCoding(object):

    def test_cyclic_code_gen_poly(self):
        code_lengths = array([15, 31])
        code_dims = array([4, 21])
        desired_genpolys = array([[2479, 3171, 3929],
                                  [1653, 1667, 1503, 1207, 1787, 1561, 1903,
                                   1219, 1137, 2013, 1453, 1897, 1975, 1395, 1547]])
        count = 0
        for n, k in zip(code_lengths, code_dims):
            genpolys = cyclic_code_genpoly(n, k)
            assert_array_equal(genpolys, desired_genpolys[count])
            count += 1
