
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

from numpy import arange, array, zeros, pi, cos, sin, sqrt, log2
from itertools import product
from commpy.utilities import bitarray2dec

class Modem:

    def modulate(self, input_bits):
        
        num_bits_symbol = int(log2(self.m))
        baseband_symbols = zeros(len(input_bits)/num_bits_symbol)*(1+1j)
        k = 0
        for i in xrange(0, len(input_bits), num_bits_symbol):
            index = bitarray2dec(input_bits[i:i+num_bits_symbol])
            baseband_symbols[k] = self.constellation[index] 
            k+=1
            
        return baseband_symbols
        
    def demodulate(self, input_symbols):
        pass


class PSKModem(Modem):
    
    def _constellation_symbol(self, i):
        return cos(2*pi*(i-1)/self.m) + sin(2*pi*(i-1)/self.m)*(0+1j)

    def __init__(self, m):
        self.m = m
        self.symbol_mapping = arange(self.m)
        self.constellation = map(self._constellation_symbol, 
                                 self.symbol_mapping)
       
class QAMModem(Modem):
    
    def _constellation_symbol(self, i):
        return (2*i[0]-1) + (2*i[1]-1)*(1j)         
        
    def __init__(self, m):
        self.m = m
        self.symbol_mapping = arange(self.m)
        mapping_array = arange(1, sqrt(self.m)+1) - (sqrt(self.m)/2)
        self.constellation = array(map(self._constellation_symbol,
                                 list(product(mapping_array, repeat=2))))
