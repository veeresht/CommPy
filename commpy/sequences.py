# Authors: CommPy contributors
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

import numpy as np
from numpy import empty, exp, pi, arange, int8, fromiter, sum

def pnsequence(pn_order, pn_seed, pn_mask, seq_length):
    """
    Generate a PN (Pseudo-Noise) sequence using a Linear Feedback Shift Register (LFSR).
    Seed and mask are ordered so that:
        - seed[-1] will be the first output
        - the new bit computed as :math:`sum(shift_register & mask) % 2` is inserted in shift[0]

    Parameters
    ----------
    pn_order : int
        Number of delay elements used in the LFSR.

    pn_seed : iterable providing 0's and 1's
        Seed for the initialization of the LFSR delay elements.
        The length of this string must be equal to 'pn_order'.

    pn_mask : iterable providing 0's and 1's
        Mask representing which delay elements contribute to the feedback
        in the LFSR. The length of this string must be equal to 'pn_order'.

    seq_length : int
        Length of the PN sequence to be generated. Usually (2^pn_order - 1)

    Returns
    -------
    pnseq : 1D ndarray of ints
        PN sequence generated.

    Raises
    ------
    ValueError
        If the pn_order is equal to the length of the strings pn_seed and pn_mask.

    """
    # Check if pn_order is equal to the length of the strings 'pn_seed' and 'pn_mask'
    if len(pn_seed) != pn_order:
        raise ValueError('pn_seed has not the same length as pn_order')
    if len(pn_mask) != pn_order:
        raise ValueError('pn_mask has not the same length as pn_order')

    # Pre-allocate memory for output
    pnseq = empty(seq_length, int8)

    # Convert input as array
    sr = fromiter(pn_seed, int8, pn_order)
    mask = fromiter(pn_mask, int8, pn_order)

    for i in range(seq_length):
        pnseq[i] = sr[-1]
        new_bit = sum(sr & mask) % 2
        sr[1:] = sr[:-1]
        sr[0] = new_bit

    return pnseq

def zcsequence(u, seq_length, q=0):
    """
    Generate a Zadoff-Chu (ZC) sequence.

    Parameters
    ----------
    u : int
        Root index of the the ZC sequence: u>0.

    seq_length : int
        Length of the sequence to be generated. Usually a prime number:
        u<seq_length, greatest-common-denominator(u,seq_length)=1.

    q : int
        Cyclic shift of the sequence (default 0).

    Returns
    -------
    zcseq : 1D ndarray of complex floats
        ZC sequence generated.
    """

    for el in [u,seq_length,q]:
        if not float(el).is_integer():
            raise ValueError('{} is not an integer'.format(el))
    if u<=0:
        raise ValueError('u is not stricly positive')
    if u>=seq_length:
        raise ValueError('u is not stricly smaller than seq_length')
    if np.gcd(u,seq_length)!=1:
        raise ValueError('the greatest common denominator of u and seq_length is not 1')

    cf = seq_length%2
    n = np.arange(seq_length)
    zcseq = np.exp( -1j * np.pi * u * n * (n+cf+2.*q) / seq_length)

    return zcseq
