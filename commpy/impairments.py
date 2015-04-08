
# Authors: Veeresh Taranalli <veeresht@gmail.com>
# License: BSD 3-Clause

"""
============================================
Impairments (:mod:`commpy.impairments`)
============================================

.. autosummary::
   :toctree: generated/

   add_frequency_offset     -- Add frequency offset impairment.

"""

from numpy import exp, pi, arange

__all__ = ['add_frequency_offset']

def add_frequency_offset(waveform, Fs, delta_f):
    """
    Add frequency offset impairment to input signal.

    Parameters
    ----------
    waveform : 1D ndarray of floats
        Input signal.

    Fs : float
        Sampling frequency (in Hz).

    delta_f : float
        Frequency offset (in Hz).

    Returns
    -------
    output_waveform : 1D ndarray of floats
        Output signal with frequency offset.
    """

    output_waveform = waveform*exp(1j*2*pi*(delta_f/Fs)*arange(len(waveform)))
    return output_waveform
