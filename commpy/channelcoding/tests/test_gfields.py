
#   Copyright 2012 Veeresh Taranalli <veeresht@gmail.com>
#
#   This file is part of CommPy.   
#
#   CommPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   CommPy is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from numpy import array, ones_like, arange
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_, assert_equal
from commpy.channelcoding.gfields import gf


class TestGaloisFields(object):

    def test_closure(self):
        for m in arange(1, 9):
            x = gf(arange(2**m), m)
            for a in x.elements:
                for b in x.elements:
                    assert_((gf(array([a]), m) + gf(array([b]), m)).elements[0] in x.elements)
                    assert_((gf(array([a]), m) * gf(array([b]), m)).elements[0] in x.elements)

    def test_addition(self):
        m = 3
        x = gf(arange(2**m), m)
        y = gf(array([6, 4, 3, 1, 2, 0, 5, 7]), m)
        z = gf(array([6, 5, 1, 2, 6, 5, 3, 0]), m)
        assert_array_equal((x+y).elements, z.elements)

    def test_multiplication(self):
        m = 3
        x = gf(array([7, 6, 5, 4, 3, 2, 1, 0]), m)
        y = gf(array([6, 4, 3, 1, 2, 0, 5, 7]), m)
        z = gf(array([4, 5, 4, 4, 6, 0, 5, 0]), m)
        assert_array_equal((x*y).elements, z.elements)


        
