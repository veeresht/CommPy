
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

""" Galois Fields """

from fractions import gcd
from numpy import array, zeros, arange, convolve, ndarray
from itertools import *
from commpy.utilities import dec2bitarray, bitarray2dec

class gf:
    """ Defines a Binary Galois Field of order m, containing n, 
    where n can be a single element or a list of elements within the field.

    Parameters
    ----------
    n : int 
    Represents the Galois field element(s).

    m : int 
    Specifies the order of the Galois Field.

    Returns
    -------
    x : int 
    A Galois Field GF(2\ :sup:`m`) object.

    Examples
    ________
    >>> from GF import gf
    >>> n = range(16)
    >>> m = 4
    >>> x = gf(n, m)
    >>> print x.elements
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    """

    # Initialization
    def __init__(self, x, m):
        self.m = m
        primpoly_array = array([0, 3, 7, 11, 19, 37, 67, 137, 285, 529, 1033, 
                                   2053, 4179, 8219, 17475, 32771, 69643])
        self.prim_poly = primpoly_array[self.m]
        if type(x) is int and x >= 0 and x < pow(2, m):
            self.elements = array([x])
        elif type(x) is ndarray and len(x) >= 1:
            self.elements = x

    # Overloading addition operator for Galois Field
    def __add__(self, x):
        if len(self.elements) == len(x.elements):
            return gf(self.elements ^ x.elements, self.m)
        else:
            raise ValueError, "The arguments should have the same number of elements"

    # Overloading multiplication operator for Galois Field
    def __mul__(self, x):
        if len(x.elements) == len(self.elements):
            prod_elements = arange(len(self.elements))
            for i in xrange(len(self.elements)):
                prod_elements[i] = polymultiply(self.elements[i], x.elements[i], self.m, self.prim_poly)
            return gf(prod_elements, self.m)
        else:
             raise ValueError, "Two sets of elements cannot be multiplied"

    def power_to_tuple(self):
        y = zeros(len(self.elements))
        for idx, i in enumerate(self.elements):
            if 2**i < 2**self.m:
                y[idx] = 2**i
            else:
                y[idx] = polydivide(2**i, self.prim_poly)
        return gf(y, self.m)

    def tuple_to_power(self):
        y = zeros(len(self.elements))
        for idx, i in enumerate(self.elements):
            if i != 0:
                init_state = 1
                cur_state = 1
                power = 0
                while cur_state != i:
                    cur_state = ((cur_state << 1) & (2**self.m-1)) ^ (-((cur_state & 2**(self.m-1)) >> (self.m - 1)) & 
                                (self.prim_poly & (2**self.m-1)))
                    power+=1
                y[idx] = power
            else:
                y[idx] = 0
        return gf(y, self.m)

    def order(self):
        orders = zeros(len(self.elements))
        power_gf = self.tuple_to_power()
        for idx, i in enumerate(power_gf.elements):
            orders[idx] = (2**self.m - 1)/(gcd(i, 2**self.m-1))

        return orders

   
# Divide two polynomials and returns the remainder
def polydivide(x, y):
    r = y
    while len(bin(r)) >= len(bin(y)):
        shift_count = len(bin(x)) - len(bin(y))
        if shift_count > 0:
            d = y << shift_count
        else:
            d = y
        x = x ^ d
        r = x
    return r

def polymultiply(x, y, m, prim_poly):
    x_array = dec2bitarray(x, m)
    y_array = dec2bitarray(y, m)
    prod = bitarray2dec(convolve(x_array, y_array) % 2)
    return polydivide(prod, prim_poly)


#def cosets(m):
#    """ Returns the cyclotomic cosets for the binary galois field. 
#    A cyclotomic coset consists of elements which are the roots of 
#    the same polynomial called the mimimal polynomial.
#
#    Parameters
#    __________
#    m : int 
#    Specifies the order of the Galois Field.
#
#    Returns
#    _______
#    co_sets : 2d list 
#    Each 1D list represents a cyclotomic coset within the 2D list.
#
#    Examples
#    ________
#    >>> print cosets(4)
#    [[1], [2, 4, 3, 5], [8, 12, 10, 15], [6, 7], [11, 14, 13, 9]]
#    """
#    N = 2**m
#    mark_list = zeros(N-1)
#    coset_no = 1
#    for i in xrange(N-1):
#        if mark_list[i] == 0:
#            j = i
#            s = 0
#            while mark_list[j] == 0:
#                mark_list[j] = coset_no
#                s += 1
#                j = (j*2) % (N-1)
#            coset_no += 1
#    co_sets = array([gf(array([1]), m)] * max(mark_list))
#    gf_array_coset = [[]* 1 for i in xrange(max(mark_list))]
#    for i in range(len(mark_list)):
#        gf_array_coset[mark_list[i]-1].append(power2tuple(i, m))
#
#    for i in range(len(gf_array_coset)):
#        co_sets[i] = gf(np.array(gf_array_coset[i]), m)
#
#    return co_sets

# Generate the minimal polynomials for elements in the Galois Field
#def minpol(gfield):
#    co_sets = cosets(gfield.m)
#    min_polynomial = [[]* 1 for i in xrange(len(co_sets))]
#    for i in range(len(co_sets)):
#        set_elements = co_sets[i].elements
#        psets = powerset(set_elements)
        #print psets
#        d = {}
#        for count in range(1, len(set_elements)+1):
#            d[count] = 0
#        for c_set in psets:
            #print len(c_set)
#            d[len(c_set)] ^= reduce(lambda x,y: polymultiply(x, y, gfield.m), c_set)
#        min_polynomial[i].append(1)
#        for key in d.keys():
#           min_polynomial[i].append(d[key])
#    minpol_list = []
#    for poly in min_polynomial:
#        val = 0
#        for i in range(len(poly)):
#            val += poly[-i-1]*pow(2, i)
#        minpol_list.append(val)
#    minpol_table = [2]*pow(2, gfield.m)
#    count = 0
#    for cset in co_sets:
#        for value in cset.elements:
#            minpol_table[value] = minpol_list[count]
#        count += 1
#
#    return minpol_table

# Convert an integer to string of bits
#def int2bin(n, count=24):
#    """returns the binary of integer n, using count number of digits"""
#    return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

# Generate the powerset given a list of elements
#def powerset(original_list):
#    list_size = len(original_list)
#    num_sets = 2**list_size
#    powerset = []
    # Don't include empty set
#    for i in range(num_sets)[1:]:
#        subset = []
#        binary_digits = list(int2bin(i, list_size))
#        list_indices = range(list_size)
#        for (bit,index) in zip(binary_digits,list_indices):
#            if bit == '1':
#                subset.append(original_list[index])
#        powerset.append(subset)
#    return powerset
