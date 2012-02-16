
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

""" Utilities module """

import numpy as np

def dec2bitarray(in_number, bit_width):
    """
    Converts a positive integer to NumPy array of the specified size containing 
    bits (0 and 1).
    
    Parameters
    ----------
    in_number : int
        Positive integer to be converted to a bit array.
    
    bit_width : int
        Size of the output bit array.
    
    Returns
    -------
    bitarray : 1D ndarray of ints
        Array containing the binary representation of the input decimal.
    
    """

    binary_string = bin(in_number)
    length = len(binary_string)
    bitarray = np.zeros(bit_width, 'int')
    for i in xrange(length-2):
        bitarray[bit_width-i-1] = int(binary_string[length-i-1]) 

    return bitarray

def bitarray2dec(in_bitarray):
    """
    Converts an input NumPy array of bits (0 and 1) to a decimal integer.

    Parameters
    ----------
    in_bitarray: 1D ndarray of ints
        Input NumPy array of bits.
    """

    number = 0

    for i in xrange(len(in_bitarray)):
        number = number + in_bitarray[i]*pow(2, len(in_bitarray)-1-i)
  
    return number

def hamming_dist(in_bitarray_1, in_bitarray_2):
    """
    Computes the Hamming distance between two NumPy arrays of bits (0 and 1).

    Parameters
    ----------
    in_bit_array_1: 1D ndarray of ints
        NumPy array of bits.

    in_bit_array_2: 1-D ndarray of ints
        NumPy array of bits.
    """

    distance = np.bitwise_xor(in_bitarray_1, in_bitarray_2).sum()
    
    return distance

def euclid_dist(in_array1, in_array2):
    """
    Computes the squared euclidean distance between two NumPy arrays

    Parameters
    ----------
    in_bitarray_1: 1-D ndarray of ints
        NumPy array of real values.

    in_bitarray_2: 1-D ndarray of ints
        NumPy array of real values.
    """
    distance = ((in_array1 - in_array2)*(in_array1 - in_array2)).sum()
        
    return distance
    


