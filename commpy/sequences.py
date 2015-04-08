
# Authors: Veeresh Taranalli <veeresht@gmail.com>
# License: BSD 3-Clause

"""
==================================================
Sequences (:mod:`commpy.sequences`)
==================================================

.. autosummary::
   :toctree: generated/

   pnsequence             -- PN Sequence Generator.
   zcsequence             -- Zadoff-Chu (ZC) Sequence Generator.

"""
__all__ = ['pnsequence', 'zcsequence']

from numpy import array, empty, zeros, roll, exp, pi, arange

def pnsequence(pn_order, pn_seed, pn_mask, seq_length):
    """
    Generate a PN (Pseudo-Noise) sequence using a Linear Feedback Shift Register (LFSR).

    Parameters
    ----------
    pn_order : int
        Number of delay elements used in the LFSR.

    pn_seed : string containing 0's and 1's
        Seed for the initialization of the LFSR delay elements.
        The length of this string must be equal to 'pn_order'.

    pn_mask : string containing 0's and 1's
        Mask representing which delay elements contribute to the feedback
        in the LFSR. The length of this string must be equal to 'pn_order'.

    seq_length : int
        Length of the PN sequence to be generated. Usually (2^pn_order - 1)

    Returns
    -------
    pnseq : 1D ndarray of ints
        PN sequence generated.

    """
    # Check if pn_order is equal to the length of the strings 'pn_seed' and 'pn_mask'

    pnseq = zeros(seq_length)

    # Initialize shift register with the pn_seed
    sr = array(map(lambda i: int(pn_seed[i]), xrange(0, len(pn_seed))))

    for i in xrange(seq_length):
        new_bit = 0
        for j in xrange(pn_order):
            if int(pn_mask[j]) == 1:
                new_bit = new_bit ^ sr[j]
        pnseq[i] = sr[pn_order-1]
        sr = roll(sr, 1)
        sr[0] = new_bit

    return pnseq.astype(int)

def zcsequence(u, seq_length):
    """
    Generate a Zadoff-Chu (ZC) sequence.

    Parameters
    ----------
    u : int
        Root index of the the ZC sequence.

    seq_length : int
        Length of the sequence to be generated. Usually a prime number.

    Returns
    -------
    zcseq : 1D ndarray of complex floats
        ZC sequence generated.
    """
    zcseq = exp((-1j * pi * u * arange(seq_length) * (arange(seq_length)+1)) / seq_length)

    return zcseq
