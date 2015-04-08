

# Authors: Veeresh Taranalli <veeresht@gmail.com>
# License: BSD 3-Clause

""" Interleavers and De-interleavers """

from numpy import array, arange, zeros
from numpy.random import mtrand

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
        rand_gen = mtrand.RandomState(seed)
        self.p_array = rand_gen.permutation(arange(length))


#class SRandInterlv(_Interleaver):


#class QPPInterlv(_Interleaver):
