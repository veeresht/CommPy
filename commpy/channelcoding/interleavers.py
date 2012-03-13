
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

__all__ = ['rand_interlv', 'rand_deinterlv']

def rand_interlv(in_array, seed):
    """ Random interleaver. 

    Parameters
    ----------
    in_array : 1D ndarray of ints
        Input data to be interleaved.

    seed : int
        Seed to initialize the random number generator 
        which generates the random permutation for 
        interleaving. 
    
    Returns
    -------
    out_array : 1D ndarray of ints
        Interleaved output data.

    Note
    ----
    The random number generator is the 
    RandomState object from NumPy, 
    which uses the Mersenne Twister algorithm. 
    
    """ 
    rand_gen = RandomState(seed)
    p_array = rand_gen.permutation(arange(len(in_array)))
    out_array = array(map(lambda x: in_array[x], p_array))

    return out_array
    
def rand_deinterlv(in_array, seed):
    """ Random de-interleaver        

    Parameters
    ----------
    in_array : 1D ndarray of ints
        Input data to be de-interleaved.

    seed : int
        Seed to initialize the random number generator 
        which generates the random permutation for interleaving. 
        This has to be the same as the seed used during 
        interleaving to obtain correctly de-interleaved data. 
    
    Returns
    -------
    out_array : 1D ndarray of ints
        De-interleaved output data.

    Note
    ----
    The random number generator is the RandomState object from NumPy, 
    which uses the Mersenne Twister algorithm. 
        
    """

    rand_gen = RandomState(seed)
    p_array = rand_gen.permutation(arange(len(in_array)))
    out_array = zeros(len(in_array), in_array.dtype)
    for index, element in enumerate(p_array):
        out_array[element] = in_array[index]
  
    return out_array
