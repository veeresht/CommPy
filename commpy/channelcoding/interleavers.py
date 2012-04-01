
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

""" Interleavers and De-interleavers """

from numpy import array, arange, zeros
from numpy.random.mtrand import RandomState

__all__ = ['RandInterlv']

class _Interleaver:

    def interlv(self, in_array):
        """ Interleave input array using the specific interleaver.

        Parameters
        ----------
        in_array : 1D ndarray of ints
            Input data to be interleaved.

        Returns
        -------
        out_array : 1D ndarray of ints
            Interleaved output data.

        """
        out_array = array(map(lambda x: in_array[x], self.p_array))
        return out_array
    
    def deinterlv(self, in_array):
        """ De-interleave input array using the specific interleaver.

        Parameters
        ----------
        in_array : 1D ndarray of ints
            Input data to be de-interleaved.

        Returns
        -------
        out_array : 1D ndarray of ints
            De-interleaved output data.

        """
        out_array = zeros(len(in_array), in_array.dtype)
        for index, element in enumerate(self.p_array):
            out_array[element] = in_array[index]
        return out_array

class RandInterlv(_Interleaver):
    """ Random Interleaver. 

    Parameters
    ----------
    length : int
        Length of the interleaver.

    seed : int
        Seed to initialize the random number generator 
        which generates the random permutation for 
        interleaving. 
    
    Returns
    -------
    random_interleaver : RandInterlv object
        A random interleaver object.

    Note
    ----
    The random number generator is the 
    RandomState object from NumPy, 
    which uses the Mersenne Twister algorithm. 
    
    """ 
    def __init__(self, length, seed):
        rand_gen = RandomState(seed)
        self.p_array = rand_gen.permutation(arange(length))


#class SRandInterlv(_Interleaver):


#class QPPInterlv(_Interleaver):



