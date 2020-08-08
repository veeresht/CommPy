# Authors: CommPy contributors
# License: BSD 3-Clause

"""
============================================
Utilities (:mod:`commpy.utilities`)
============================================

.. autosummary::
   :toctree: generated/

   dec2bitarray         -- Integer or array-like of integers to binary (bit array).
   decimal2bitarray     -- Specialized version for one integer to binary (bit array).
   bitarray2dec         -- Binary (bit array) to integer.
   hamming_dist         -- Hamming distance.
   euclid_dist          -- Squared Euclidean distance.
   upsample             -- Upsample by an integral factor (zero insertion).
   signal_power         -- Compute the power of a discrete time signal.
"""
import functools

import numpy as np

__all__ = ['dec2bitarray', 'decimal2bitarray', 'bitarray2dec', 'hamming_dist', 'euclid_dist', 'upsample',
           'signal_power']

vectorized_binary_repr = np.vectorize(np.binary_repr)


def dec2bitarray(in_number, bit_width):
    """
    Converts a positive integer or an array-like of positive integers to NumPy array of the specified size containing
    bits (0 and 1).

    Parameters
    ----------
    in_number : int or array-like of int
        Positive integer to be converted to a bit array.

    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D ndarray of numpy.int8
        Array containing the binary representation of all the input decimal(s).

    """

    if isinstance(in_number, (np.integer, int)):
        return decimal2bitarray(in_number, bit_width).copy()
    result = np.zeros(bit_width * len(in_number), np.int8)
    for pox, number in enumerate(in_number):
        result[pox * bit_width:(pox + 1) * bit_width] = decimal2bitarray(number, bit_width).copy()
    return result


@functools.lru_cache(maxsize=128, typed=False)
def decimal2bitarray(number, bit_width):
    """
    Converts a positive integer to NumPy array of the specified size containing bits (0 and 1). This version is slightly
    quicker that dec2bitarray but only work for one integer.

    Parameters
    ----------
    in_number : int
        Positive integer to be converted to a bit array.

    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D ndarray of numpy.int8
        Array containing the binary representation of all the input decimal(s).

    """
    result = np.zeros(bit_width, np.int8)
    i = 1
    pox = 0
    while i <= number:
        if i & number:
            result[bit_width - pox - 1] = 1
        i <<= 1
        pox += 1
    return result


def bitarray2dec(in_bitarray):
    """
    Converts an input NumPy array of bits (0 and 1) to a decimal integer.

    Parameters
    ----------
    in_bitarray : 1D ndarray of ints
        Input NumPy array of bits.

    Returns
    -------
    number : int
        Integer representation of input bit array.
    """

    number = 0

    for i in range(len(in_bitarray)):
        number = number + in_bitarray[i] * pow(2, len(in_bitarray) - 1 - i)

    return number


def hamming_dist(in_bitarray_1, in_bitarray_2):
    """
    Computes the Hamming distance between two NumPy arrays of bits (0 and 1).

    Parameters
    ----------
    in_bit_array_1 : 1D ndarray of ints
        NumPy array of bits.

    in_bit_array_2 : 1D ndarray of ints
        NumPy array of bits.

    Returns
    -------
    distance : int
        Hamming distance between input bit arrays.
    """

    distance = np.bitwise_xor(in_bitarray_1, in_bitarray_2).sum()

    return distance


def euclid_dist(in_array1, in_array2):
    """
    Computes the squared euclidean distance between two NumPy arrays

    Parameters
    ----------
    in_array1 : 1D ndarray of floats
        NumPy array of real values.

    in_array2 : 1D ndarray of floats
        NumPy array of real values.

    Returns
    -------
    distance : float
        Squared Euclidean distance between two input arrays.
    """
    distance = ((in_array1 - in_array2) * (in_array1 - in_array2)).sum()

    return distance


def upsample(x, n):
    """
    Upsample the input array by a factor of n

    Adds n-1 zeros between consecutive samples of x

    Parameters
    ----------
    x : 1D ndarray
        Input array.

    n : int
        Upsampling factor

    Returns
    -------
    y : 1D ndarray
        Output upsampled array.
    """
    y = np.empty(len(x) * n, dtype=complex)
    y[0::n] = x
    zero_array = np.zeros(len(x), dtype=complex)
    for i in range(1, n):
        y[i::n] = zero_array

    return y


def signal_power(signal):
    """
    Compute the power of a discrete time signal.

    Parameters
    ----------
    signal : 1D ndarray
             Input signal.

    Returns
    -------
    P : float
        Power of the input signal.
    """

    @np.vectorize
    def square_abs(s):
        return abs(s) ** 2

    P = np.mean(square_abs(signal))
    return P
