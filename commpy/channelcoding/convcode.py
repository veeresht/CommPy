
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

# =============================================================================
#   HELPER FUNCTION
#   Convert a decimal number to a NumPy array of bits of size M
# =============================================================================
def _dec_2_bit_array(in_number, bit_width):
    
    binary_string = bin(in_number)
    bitarray = np.zeros(bit_width, 'int')

    for i in range(len(binary_string)-2):
        bitarray[bit_width-i-1] = int(binary_string[len(binary_string)-i-1]) 

    return bitarray

# =============================================================================
#   HELPER FUNCTION
#   Convert a NumPy array of bits to a decimal number
# =============================================================================
def _bit_array_2_dec(in_bitarray):

    number = 0

    for i in range(len(in_bitarray)):
        number = number + in_bitarray[i]*pow(2, len(in_bitarray)-1-i)
  
    return number

# =============================================================================
#   HELPER FUNCTION
#   Compute the Hamming distance between two Bit Arrays
# =============================================================================
def hamming_dist(in_bitarray_1, in_bitarray_2):

    distance = np.sum(np.bitwise_xor(in_bitarray_1, in_bitarray_2))
    
    return distance

# =============================================================================
#   Generates the next_state_table and the output_table for the encoder/decoder
# =============================================================================
def _generate_tables(generator_matrix, M):
    
    # =========================================================================
    # generator_matrix - NumPy array (2D) containing the polynomials 
    #                    corresponding to each output bit
    # M ---------------- Memory of the encoder (shift register)
    # =========================================================================

    # Derive the encoder parameters using the G(D) matrix
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

    # Initialize the shift register
    #shift_register = np.zeros(M, 'int')
    
    # Compute the entries in the next state table and the output table
    # Loop over all possible states
    for current_state in range(number_states):

        #shift_register = np.zeros(M[current_state], 'int')
       
        # Loop over all possible inputs
        for current_input in range(number_inputs):
            #shift_register = np.zeros(M[current_input], 'int')
            
            # Initialize the shift register array to represent the current_state
            #shift_register = _dec_2_bit_array(current_state, M[current_input])
            # Initialize the array of output bits of dimension n
            outbits = np.zeros(n, 'int')
            
            # Compute the values in the output_table
            # Loop over n outputs
            for r in range(n):
                
                output_generator_array = np.zeros(k, 'int')
                shift_register = _dec_2_bit_array(current_state, M.sum())

                for l in range(k):
                    #print M[l], current_state
                                       
                    # Convert the number representing a polynomial into a bit array
                    generator_array = _dec_2_bit_array(generator_matrix[l][r], M[l]+1)

                    # Loop over M delay elements of the shift register 
                    # to compute their contribution to the r-th output
                    for i in range(M[l]):
                        outbits[r] = (outbits[r] + \
                            (shift_register[i+l]*generator_array[i+1]))%2
                        
                        #next_shift_register[1:]

                    output_generator_array[l] = generator_array[0]
                    if l == 0:
                        shift_register[1:M[l]] = shift_register[0:M[l]-1]
                        shift_register[0] = _dec_2_bit_array(current_input, k)[0]
                    else:
                        shift_register[l+M[l-1]:l+M[l-1]+M[l]-1] = \
                                shift_register[l+M[l-1]-1:l+M[l-1]+M[l]-2] 
                        shift_register[l+M[l-1]-1] = \
                                _dec_2_bit_array(current_input, k)[l]

                # Compute the contribution of the current_input to the output
                outbits[r] = (outbits[r] + \
                    (np.sum(_dec_2_bit_array(current_input, k) * \
                    output_generator_array)%2))%2
            
            # Update the ouput_table using the computed output value
            output_table[current_state][current_input] = \
                _bit_array_2_dec(outbits)
            
            # Shift the bits in the shift register by '1' (Clock Cycle)
            #shift_register[1:] = shift_register[0:M[current_input]-1]

            # Read in the input bit into the shift register
            #shift_register[0] = current_input

            # Update the next_state_table using the new state of 
            # the shift register
            next_state_table[current_state][current_input] = \
                _bit_array_2_dec(shift_register)
    
    #print next_state_table
    #print output_table

    return [next_state_table, output_table]


def convencode(message_bits, generator_matrix, M):
    """
    Convolutionally encode message bits using by specifying a generator matrix 
    G(D) for the code and the memory of the encoder.

    Parameters
    ----------
    message_bits : 1-D ndarray
        Stream of bits to be convolutionally encoded.
    generator_matrix : 2-D ndarray
        Generator matrix G(D) of the convolutional code using which the input 
        bits are to be encoded
    M : int
        Memory of the convolutional encoder (shift register)
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
    for i in range(number_inbits/k-1): # Loop through all input bits
        current_input = _bit_array_2_dec(inbits[i*k:(i+1)*k])
        current_output = output_table[current_state][current_input]
        outbits[j*n:(j+1)*n] = _dec_2_bit_array(current_output, n)
        current_state = next_state_table[current_state][current_input]
        j = j + 1
   
    return outbits


def viterbi_decode(coded_bits, generator_matrix, M, tb_depth):
    """
    Decodes a stream of convolutionally encoded bits using the Viterbi Algorithm

    Parameters
    ----------
    coded_bits : 1-D ndarray
        Stream of convolutionally encoded bits which are to be decoded.
    generator_matrix : 2-D ndarray
        Generator matrix G(D) of the convolutional code using which the input 
        bits are to be decoded.
    M : int
        Memory of the convolutional encoder (shift register)
    tb_length : int
        Traceback depth (Typically set to 5*(M+1))

    References
    ----------
    [1]. Todd K. Moon. Error Correction Coding: Mathematical Methods and 
    Algorithms. John Wiley and Sons, 2005.
    """

    # Derive the encoder parameters using the G(D) matrix
    # k = Rows in G(D), n = columns in G(D)
    [k, n] = generator_matrix.shape
    rate = float(k)/n

    # Compute the number of states in the encoder using
    # the number of memory elements
    number_states = pow(2, M.sum())
    
    # Compute the number of input symbols (depends on k)
    number_inputs = pow(2, k)

    L = int(len(coded_bits)*rate)

    [next_state_table, output_table] = _generate_tables(generator_matrix, M)
    
    path_metrics = np.zeros([number_states, 2])
    path_metrics[:,:] = np.inf
    path_metrics[0][0] = 0
    
    paths = np.zeros([number_states, tb_depth])
    paths[:,:] = np.nan
    paths[0][0] = 0

    decoded_symbols = np.zeros([number_states, tb_depth], 'int')
    
    decoded_bits = np.zeros(L+2*tb_depth-1, 'int')

    t = 1
    tb_count = 1
    count = 0
    while t < (L+M.sum()+M.sum()%k)/k: #+M.sum()+M.sum()%k:
        # Get the received codeword corresponding to t
        r_codeword = _bit_array_2_dec(coded_bits[(t-1)*n:t*n])
        
        # Loop over all the current states (Time instant: t)
        for state_num in range(number_states):
            # Using the next state table find the previous states and inputs
            # leading into the current state (Trellis)
            [previous_states, inputs] = np.where(next_state_table == state_num)

            # Initialize a temporary array to store all the path metrics 
            # for the current state
            pmetrics = np.zeros(len(previous_states))
            
            # Loop over all the previous states (Time instant: t-1)
            for i in range(len(previous_states)):
                # Using the output table, find the ideal codeword 
                i_codeword = output_table[previous_states[i]][inputs[i]]

                # Compute the branch metric (hamming distance) between 
                # the received and the ideal codeword
                branch_metric = hamming_dist( \
                        _dec_2_bit_array(r_codeword, n), \
                        _dec_2_bit_array(i_codeword, n))
                
                # ADD operation: Add the branch metric to the 
                # accumulated path metric and store it in the temporary array 
                pmetrics[i] = path_metrics[previous_states[i]][0] + branch_metric
            
            # Execute COMPARE and SELECT operations only for finite values 
            # in pmetrics 
            if np.isfinite(pmetrics.min()):       
                # Compare and Select the minimum accumulated path metric
                path_metrics[state_num][1] = pmetrics.min()
                
                # Store the previous state corresponding to the minimum 
                # accumulated path metric
                paths[state_num][tb_count] = previous_states[pmetrics.argmin()]
                
                # Store the previous input corresponding to the minimum 
                # accumulated path metric
                decoded_symbols[state_num][tb_count] = inputs[pmetrics.argmin()]
        

        if t >= tb_depth - 1:
            j = tb_depth - 1
            current_state = path_metrics[:,1].argmin()
            
            # Traceback Loop
            while j >= 0:
                previous_state = int(paths[current_state][j])
                decoded_bits[t+(j)*k+count:t+(j+1)*k+count] =  \
                        _dec_2_bit_array(decoded_symbols[previous_state][j], k)
                j = j - 1
                current_state = previous_state
            
            count = count + k-1
            tb_count = tb_depth - 1
            paths[:,0:tb_depth-1] = paths[:,1:]
            decoded_symbols[:,0:tb_depth-1] = decoded_symbols[:,1:]

        else:
            decoded_bits[(t-1)*k:t*k] = _dec_2_bit_array(0, k)
            tb_count = tb_count + 1

        # Increment time
        t = t + 1 

        # Path metrics (at t-1) = Path metrics (at t)
        path_metrics[:,0] = path_metrics[:,1]

        # Force all the paths back to '0' state at the end of decoding
        if t == (L+M.sum()+M.sum()%k)/k-1:
            number_states = 1
     
    return decoded_bits[M.sum()+M.sum()%k-1:len(decoded_bits)-tb_depth]

