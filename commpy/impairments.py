
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
