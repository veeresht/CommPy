
#   Copyright 2012 Veeresh Taranalli <veeresht@gmail.com>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from commpy.utilities import *
from commpy.channelcoding.acstb import *


__all__ = ['convencode', 'viterbi_decode']



# =============================================================================
#   Generates the next_state_table and the output_table for the encoder/decoder
# =============================================================================
def _generate_tables(generator_matrix, M):
    
    # =========================================================================
    # generator_matrix - NumPy array (2D) containing the polynomials 
    #                    corresponding to each output bit
    # M ---------------- Memory of the encoder (shift register)
    # =========================================================================

    # Derive the encoder parameters using the G(D) matrix)
    [k, n] = generator_matrix.shape
    
    # Compute the number of states in the encoder using
    # the number of memory elements
    number_states = pow(2, M.sum())
    
    # Compute the number of input symbols (depends on k)
    number_inputs = pow(2, k)

    # Initialize the next state table
    next_state_table = np.zeros([number_states, number_inputs], 'int')

    # Initialize the output table
    output_table = np.zeros([number_states, number_inputs], 'int')
    
    # Initialize the input table
    input_table = np.zeros([number_states, number_states], 'int')
    
    # Compute the entries in the next state table and the output table
    # Loop over all possible states
    for current_state in xrange(number_states):
       
        # Loop over all possible inputs
        for current_input in xrange(number_inputs):
            # Initialize the array of output bits of dimension n
            outbits = np.zeros(n, 'int')
            
            # Compute the values in the output_table
            # Loop over n outputs
            for r in xrange(n):
                
                output_generator_array = np.zeros(k, 'int')
                shift_register = dec2bitarray(current_state, M.sum())

                for l in xrange(k):
                                       
                    # Convert the number representing a polynomial into a bit array
                    generator_array = dec2bitarray(generator_matrix[l][r], M[l]+1)

                    # Loop over M delay elements of the shift register 
                    # to compute their contribution to the r-th output
                    for i in xrange(M[l]):
                        outbits[r] = (outbits[r] + \
                            (shift_register[i+l]*generator_array[i+1]))%2

                    output_generator_array[l] = generator_array[0]
                    if l == 0:
                        shift_register[1:M[l]] = shift_register[0:M[l]-1]
                        shift_register[0] = dec2bitarray(current_input, k)[0]
                    else:
                        shift_register[l+M[l-1]:l+M[l-1]+M[l]-1] = \
                                shift_register[l+M[l-1]-1:l+M[l-1]+M[l]-2] 
                        shift_register[l+M[l-1]-1] = \
                                dec2bitarray(current_input, k)[l]

                # Compute the contribution of the current_input to the output
                outbits[r] = (outbits[r] + \
                    (np.sum(dec2bitarray(current_input, k) * \
                    output_generator_array)%2))%2
            
            # Update the ouput_table using the computed output value
            output_table[current_state][current_input] = \
                bitarray2dec(outbits)

            # Update the next_state_table using the new state of 
            # the shift register
            next_state_table[current_state][current_input] = \
                bitarray2dec(shift_register)

    return [next_state_table, output_table]


def convencode(message_bits, generator_matrix, M):
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
    
    # Derive the encoder parameters using the G(D) matrix
    # k = Rows in G(D), n = Columns in G(D)
    [k, n] = generator_matrix.shape
    rate = float(k)/n

    # Store the number of message bits to be encoded
    number_message_bits = np.size(message_bits)

    # Initialize an array to contain the message bits plus the truncation zeros
    inbits = np.zeros(number_message_bits + M.sum() + M.sum()%k, 'int')
    number_inbits = number_message_bits + M.sum() + M.sum()%k
    
    # Pad the input bits with M zeros (L-th terminated truncation)
    inbits[0:number_message_bits] = message_bits
    
    # Compute the number of outbits to be generated
    number_outbits = int(number_inbits/rate)

    # Initialize the outbits array upfront 
    outbits = np.zeros(number_outbits, 'int')
    
    # Generate the next_state_table and the output_table
    [next_state_table, output_table] = _generate_tables(generator_matrix, M)

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


def viterbi_decode(coded_bits, generator_matrix, M, tb_depth=None, decoding_type='hard'):
    """
    Decodes a stream of convolutionally encoded bits using the Viterbi Algorithm

    Parameters
    ----------
    coded_bits : 1D ndarray 
        Stream of convolutionally encoded bits which are to be decoded.
    
    generator_matrix : 2D ndarray of ints
        Generator matrix G(D) of the convolutional code using which the input bits are to be decoded.
    
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
    
    # Derive the encoder parameters using the G(D) matrix
    # k = Rows in G(D), n = columns in G(D)
    [k, n] = generator_matrix.shape
    rate = float(k)/n
    
    # Compute the total memory for the decoder
    total_memory = M.sum()

    if tb_depth is None:
        tb_depth = 5*total_memory

    # Compute the number of states in the decoder using
    # the number of memory elements
    number_states = pow(2, total_memory)
    
    # Compute the number of input symbols (depends on k)
    number_inputs = pow(2, k)
    
    # Number of message bits after decoding
    L = int(len(coded_bits)*rate)
    
    [next_state_table, output_table] = _generate_tables(generator_matrix, M)

    path_metrics = np.empty([number_states, 2])
    path_metrics[:,:] = 10000
    path_metrics[0][0] = 0
    
    paths = np.empty([number_states, tb_depth], 'int')
    paths[:,:] = 10000
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
        
        acs_traceback(number_states, next_state_table, output_table, r_codeword, 
                decoding_type, path_metrics, paths, decoded_symbols, decoded_bits, 
                tb_count, n, k, number_inputs, t, count, tb_depth, current_number_states)        

        if t >= tb_depth - 1:
            tb_count = tb_depth - 1
            count = count + k - 1
        else:
            tb_count = tb_count + 1

        # Path metrics (at t-1) = Path metrics (at t)
        path_metrics[:,0] = path_metrics[:,1]

        # Force all the paths back to '0' state at the end of decoding
        if t == (L+total_memory+total_memory%k)/k:
            current_number_states = 1
            
    return decoded_bits[0:len(decoded_bits)-tb_depth-M.sum()-M.sum()%k-k]

