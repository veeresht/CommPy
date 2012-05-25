
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
Modulation Demodulation (:mod:`commpy.modulation`)
============================================

.. autosummary::
   :toctree: generated/
    
   PSKModem             -- Phase Shift Keying (PSK) Modem.
   QAMModem             -- Quadrature Amplitude Modulation (QAM) Modem.

"""
from numpy import arange, array, zeros, pi, cos, sin, sqrt, log2, argmin, hstack
from itertools import product
from commpy.utilities import bitarray2dec, dec2bitarray

__all__ = ['PSKModem', 'QAMModem']

class Modem:

    def modulate(self, input_bits):
        """ Modulate (map) an array of bits to constellation symbols.

        Parameters
        ----------
        input_bits : 1D ndarray of ints
            Inputs bits to be modulated (mapped).

        Returns
        -------
        baseband_symbols : 1D ndarray of complex floats
            Modulated complex symbols.
        
        """
        num_bits_symbol = int(log2(self.m))
        index_list = map(lambda i: bitarray2dec(input_bits[i:i+num_bits_symbol]), \
                         xrange(0, len(input_bits), num_bits_symbol))
        baseband_symbols = self.constellation[index_list]

        return baseband_symbols
        
    def demodulate(self, input_symbols):
        """ Demodulate (map) a set of constellation symbols to corresponding bits.
        
        Supports hard-decision demodulation only.

        Parameters
        ----------
        input_symbols : 1D ndarray of complex floats
            Input symbols to be demodulated.

        Returns
        -------
        demod_bits : 1D ndarray of ints
            Corresponding demodulated bits.
            
        """
        num_bits_symbol = int(log2(self.m))
        index_list = map(lambda i: argmin(abs(input_symbols[i] - self.constellation)), \
                         xrange(0, len(input_symbols)))
        demod_bits = hstack(map(lambda i: dec2bitarray(i, num_bits_symbol), index_list))

        return demod_bits


class PSKModem(Modem):
    """ Creates a Phase Shift Keying (PSK) Modem object. """

    def _constellation_symbol(self, i):
        return cos(2*pi*(i-1)/self.m) + sin(2*pi*(i-1)/self.m)*(0+1j)

    def __init__(self, m):
        """ Creates a Phase Shift Keying (PSK) Modem object.

        Parameters
        ----------
        m : int
            Size of the PSK constellation.
        
        """
        self.m = m
        self.symbol_mapping = arange(self.m)
        self.constellation = map(self._constellation_symbol, 
                                 self.symbol_mapping)
       
class QAMModem(Modem):
    """ Creates a Quadrature Amplitude Modulation (QAM) Modem object."""
    
    def _constellation_symbol(self, i):
        return (2*i[0]-1) + (2*i[1]-1)*(1j)         
        
    def __init__(self, m):
        """ Creates a Quadrature Amplitude Modulation (QAM) Modem object.

        Parameters
        ----------
        m : int
            Size of the QAM constellation.

        """

        self.m = m
        self.symbol_mapping = arange(self.m)
        mapping_array = arange(1, sqrt(self.m)+1) - (sqrt(self.m)/2)
        self.constellation = array(map(self._constellation_symbol,
                                 list(product(mapping_array, repeat=2))))
