
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

""" Algorithms for Convolutional Codes """

import numpy as np
from commpy.utilities import dec2bitarray, bitarray2dec
from commpy.channelcoding.acstb import acs_traceback

__all__ = ['Trellis', 'conv_encode', 'viterbi_decode']

class Trellis:
    """
    Class defining a Trellis corresponding to a k/n - rate convolutional code.
    
    Parameters
    ----------
    memory : 1D ndarray of ints
        Number of memory elements per input of the convolutional encoder.

    g_matrix : 2D ndarray of ints (octal representation)
        Generator matrix G(D) of the convolutional encoder. Each element of 
        G(D) represents a polynomial. 

    feedback : int, optional
        Feedback polynomial of the convolutional encoder. Default value is 00.
        
    code_type : {'default', 'rsc'}, optional
        Use 'rsc' to generate a recursive systematic convolutional code. 
        If 'rsc' is specified, then the first 'k x k' sub-matrix of G(D) must 
        represent a identity matrix along with a non-zero feedback polynomial.
    
    
    Attributes
    ----------
    k : int 
        Size of the smallest block of input bits that can be encoded using 
        the convolutional code.

    n : int
        Size of the smallest block of output bits generated using 
        the convolutional code.

    total_memory : int
        Total number of delay elements needed to implement the convolutional 
        encoder.

    number_states : int
        Number of states in the convolutional code trellis.

    number_inputs: int
        Number of branches from each state in the convolutional code trellis.

    next_state_table : 2D ndarray of ints
        Table representing the state transition matrix of the 
        convolutional code trellis. Rows represent current states and 
        columns represent current inputs in decimal. Elements represent the 
        corresponding next states in decimal.
 
    output_table:
        Table representing the output matrix of the convolutional code trellis.
        Rows represent current states and columns represent current inputs in 
        decimal. Elements represent corresponding outputs in decimal.
    
    Examples
    --------
    >>> from numpy import array
    >>> import commpy.channelcoding.convcode as cc
    >>> memory = array([2])
    >>> g_matrix = array([[05, 07]]) # G(D) = [1+D^2, 1+D+D^2]
    >>> trellis = cc.Trellis(memory, g_matrix)
    >>> print trellis.k
    1
    >>> print trellis.n
    2
    >>> print trellis.total_memory
    2
    >>> print trellis.number_states
    4
    print trellis.number_inputs
    2
    >>> print trellis.next_state_table
    [[0 2]
     [0 2]
     [1 3]
     [1 3]]
    >>>print trellis.output_table
    [[0 3]
     [3 0]
     [1 2]
     [2 1]]
    
    """
    def __init__(self, memory, g_matrix, feedback = 0, code_type = 'default'):
        
        [self.k, self.n] = g_matrix.shape
        
        if code_type == 'rsc':
            for i in xrange(self.k):
                g_matrix[i][i] = feedback

        self.total_memory = memory.sum()
        self.number_states = pow(2, self.total_memory)
        self.number_inputs = pow(2, self.k) 
        self.next_state_table = np.zeros([self.number_states, 
                                          self.number_inputs], 'int')
        self.output_table = np.zeros([self.number_states, 
                                      self.number_inputs], 'int')
    
        # Compute the entries in the next state table and the output table
        for current_state in xrange(self.number_states):
       
            for current_input in xrange(self.number_inputs):
                outbits = np.zeros(self.n, 'int')
            
                # Compute the values in the output_table
                for r in xrange(self.n):
                
                    output_generator_array = np.zeros(self.k, 'int')
                    shift_register = dec2bitarray(current_state, 
                                                  self.total_memory)

                    for l in xrange(self.k):
                                       
                        # Convert the number representing a polynomial into a 
                        # bit array
                        generator_array = dec2bitarray(g_matrix[l][r], 
                                                       memory[l]+1)

                        # Loop over M delay elements of the shift register 
                        # to compute their contribution to the r-th output
                        for i in xrange(memory[l]):
                            outbits[r] = (outbits[r] + \
                                (shift_register[i+l]*generator_array[i+1])) % 2

                        output_generator_array[l] = generator_array[0]
                        if l == 0:
                            feedback_array = (dec2bitarray(feedback, memory[l]) * shift_register[0:memory[l]]).sum()
                            shift_register[1:memory[l]] = \
                                    shift_register[0:memory[l]-1]
                            shift_register[0] = (dec2bitarray(current_input, 
                                    self.k)[0] + feedback_array) % 2
                        else:
                            feedback_array = (dec2bitarray(feedback, memory[l]) * 
                                    shift_register[l+memory[l-1]-1:l+memory[l-1]+memory[l]-1]).sum()
                            shift_register[l+memory[l-1]:l+memory[l-1]+memory[l]-1] = \
                                    shift_register[l+memory[l-1]-1:l+memory[l-1]+memory[l]-2] 
                            shift_register[l+memory[l-1]-1] = \
                                    (dec2bitarray(current_input, self.k)[l] + feedback_array) % 2

                    # Compute the contribution of the current_input to output
                    outbits[r] = (outbits[r] + \
                        (np.sum(dec2bitarray(current_input, self.k) * \
                        output_generator_array + feedback_array) % 2)) % 2
            
                # Update the ouput_table using the computed output value
                self.output_table[current_state][current_input] = \
                    bitarray2dec(outbits)

                # Update the next_state_table using the new state of 
                # the shift register
                self.next_state_table[current_state][current_input] = \
                    bitarray2dec(shift_register)


def conv_encode(message_bits, trellis):
    """
    Encode bits using a convolutional code.

    Parameters
    ----------  
    message_bits : 1D ndarray containing {0, 1}
        Stream of bits to be convolutionally encoded.
    
    generator_matrix : 2-D ndarray of ints
        Generator matrix G(D) of the convolutional code using which the input 
        bits are to be encoded.
    
    M : 1D ndarray of ints 
        Number of memory elements per input of the convolutional encoder.

    Returns
    -------
    coded_bits : 1D ndarray containing {0, 1} 
        Encoded bit stream.
    """  

    k = trellis.k
    n = trellis.n
    total_memory = trellis.total_memory
    rate = float(k)/n

    number_message_bits = np.size(message_bits)

    # Initialize an array to contain the message bits plus the truncation zeros
    inbits = np.zeros(number_message_bits + total_memory + total_memory % k, 
                      'int')
    number_inbits = number_message_bits + total_memory + total_memory % k
    
    # Pad the input bits with M zeros (L-th terminated truncation)
    inbits[0:number_message_bits] = message_bits
    
    number_outbits = int(number_inbits/rate) 
    outbits = np.zeros(number_outbits, 'int')
    next_state_table = trellis.next_state_table
    output_table = trellis.output_table

    # Encoding process - Each iteration of the loop represents one clock cycle
    current_state = 0
    j = 0

    for i in xrange(number_inbits/k): # Loop through all input bits
        current_input = bitarray2dec(inbits[i*k:(i+1)*k])
        current_output = output_table[current_state][current_input]
        outbits[j*n:(j+1)*n] = dec2bitarray(current_output, n)
        current_state = next_state_table[current_state][current_input]
        j = j + 1
   
    return outbits


def viterbi_decode(coded_bits, trellis, tb_depth=None, decoding_type='hard'):
    """
    Decodes a stream of convolutionally encoded bits using the Viterbi Algorithm

    Parameters
    ----------
    coded_bits : 1D ndarray 
        Stream of convolutionally encoded bits which are to be decoded.
    
    generator_matrix : 2D ndarray of ints
        Generator matrix G(D) of the convolutional code using which the 
        input bits are to be decoded.
    
    M : 1D ndarray of ints
        Number of memory elements per input of the convolutional encoder.
    
    tb_length : int
        Traceback depth (Typically set to 5*(M+1)).
    
    decoding_type : str {'hard', 'soft', 'unquantized'}
        The type of decoding to be used.
        
    Returns
    -------
    decoded_bits : 1D ndarray 
        Decoded bit stream.
    
    References
    ----------
    .. [1] Todd K. Moon. Error Correction Coding: Mathematical Methods and 
        Algorithms. John Wiley and Sons, 2005.
    """
    
    # k = Rows in G(D), n = columns in G(D)
    k = trellis.k
    n = trellis.n
    rate = float(k)/n
    total_memory = trellis.total_memory
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs

    if tb_depth is None:
        tb_depth = 5*total_memory

    next_state_table = trellis.next_state_table
    output_table = trellis.output_table
    
    # Number of message bits after decoding
    L = int(len(coded_bits)*rate)
    
    path_metrics = np.empty([number_states, 2])
    path_metrics[:, :] = 10000
    path_metrics[0][0] = 0
    paths = np.empty([number_states, tb_depth], 'int')
    paths[:, :] = 10000
    paths[0][0] = 0

    decoded_symbols = np.zeros([number_states, tb_depth], 'int')
    decoded_bits = np.empty(L+tb_depth+k, 'int')
    r_codeword = np.empty(n, 'int')

    tb_count = 1
    count = 0
    current_number_states = number_states

    for t in xrange(1, (L+total_memory+total_memory%k)/k + 1):
        # Get the received codeword corresponding to t
        if t <= L:
            r_codeword = coded_bits[(t-1)*n:t*n]
        else:
            if decoding_type == 'hard':
                r_codeword[:] = 0
            elif decoding_type == 'soft':
                pass
            elif decoding_type == 'unquantized':
                r_codeword[:] = 0
                r_codeword = r_codeword - 1
            else:
                pass
        
        acs_traceback(r_codeword, trellis, decoding_type, path_metrics, paths, 
                decoded_symbols, decoded_bits, tb_count, t, count, tb_depth, 
                current_number_states)        

        if t >= tb_depth - 1:
            tb_count = tb_depth - 1
            count = count + k - 1
        else:
            tb_count = tb_count + 1

        # Path metrics (at t-1) = Path metrics (at t)
        path_metrics[:, 0] = path_metrics[:, 1]

        # Force all the paths back to '0' state at the end of decoding
        if t == (L+total_memory+total_memory%k)/k:
            current_number_states = 1
            
    return decoded_bits[0:len(decoded_bits)-tb_depth-total_memory-total_memory%k-k]
