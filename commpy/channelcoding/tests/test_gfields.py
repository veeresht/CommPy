
# Authors: Veeresh Taranalli <veeresht@gmail.com>
# License: BSD 3 clause

from numpy import array, ones_like, arange
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_, assert_equal
from commpy.channelcoding.gfields import GF


class TestGaloisFields(object):

    def test_closure(self):
        for m in arange(1, 9):
            x = GF(arange(2**m), m)
            for a in x.elements:
                for b in x.elements:
                    assert_((GF(array([a]), m) + GF(array([b]), m)).elements[0] in x.elements)
                    assert_((GF(array([a]), m) * GF(array([b]), m)).elements[0] in x.elements)

    def test_addition(self):
        m = 3
        x = GF(arange(2**m), m)
        y = GF(array([6, 4, 3, 1, 2, 0, 5, 7]), m)
        z = GF(array([6, 5, 1, 2, 6, 5, 3, 0]), m)
        assert_array_equal((x+y).elements, z.elements)

    def test_multiplication(self):
        m = 3
        x = GF(array([7, 6, 5, 4, 3, 2, 1, 0]), m)
        y = GF(array([6, 4, 3, 1, 2, 0, 5, 7]), m)
        z = GF(array([4, 5, 4, 4, 6, 0, 5, 0]), m)
        assert_array_equal((x*y).elements, z.elements)

    def test_tuple_form(self):
        m = 3
        x = GF(arange(0, 2**m-1), m)
        y = x.power_to_tuple()
        z = GF(array([1, 2, 4, 3, 6, 7, 5]), m)
        assert_array_equal(y.elements, z.elements)

    def test_power_form(self):
        m = 3
        x = GF(arange(1, 2**m), m)
        y = x.tuple_to_power()
        z = GF(array([0, 1, 3, 2, 6, 4, 5]), m)
        assert_array_equal(y.elements, z.elements)
        m = 4
        x = GF(arange(1, 2**m), m)
        y = x.tuple_to_power()
        z = GF(array([0, 1, 4, 2, 8, 5, 10, 3, 14, 9, 7, 6, 13, 11, 12]), m)
        assert_array_equal(y.elements, z.elements)

    def test_order(self):
        m = 4
        x = GF(arange(1, 2**m), m)
        y = x.order()
        z = array([1, 15, 15, 15, 15, 3, 3, 5, 15, 5, 15, 5, 15, 15, 5])
        assert_array_equal(y, z)

    def test_minpols(self):
        m = 4
        x = GF(arange(2**m), m)
        z = array([2, 3, 19, 19, 19, 19, 7, 7, 31, 25, 31, 25, 31, 25, 25, 31])
        assert_array_equal(x.minpolys(), z)
        m = 6
        x = GF(array([2, 8, 32, 6, 24, 35, 10, 40, 59, 41, 14, 37]), m)
        z = array([67, 87, 103, 73, 13, 109, 91, 117, 7, 115, 11, 97])
        assert_array_equal(x.minpolys(), z)
