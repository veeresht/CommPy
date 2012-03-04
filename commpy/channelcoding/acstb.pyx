
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

import numpy as np

cimport numpy as np
cimport cython

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void dec2bitarray_c(int in_number, int bit_width, np.ndarray[np.int_t, ndim=1] bitarray):
    
    cdef int i
    
    for i in xrange(bit_width):
        bitarray[bit_width-i-1] = (in_number & 1)
        in_number = in_number >> 1

cdef int hamming_dist_c(np.ndarray[np.int_t, ndim=1] in_bitarray_1, np.ndarray[np.int_t, ndim=1] in_bitarray_2, int length):
    
    cdef int distance = 0, i
    for i in xrange(length):
        distance = distance + (in_bitarray_1[i] ^ in_bitarray_2[i])

    return distance

cdef float euclid_dist_c(np.ndarray inarray1, np.ndarray inarray2, int length):

    cdef int i
    cdef float distance = 0
    for i in xrange(length):
        distance = distance + ((inarray1[i] - inarray2[i])*(inarray1[i] - inarray2[i]))

    return distance

cdef int where_c(np.ndarray[np.int_t, ndim=2] inarray, int rows, int cols, int search_value, np.ndarray[np.int_t, ndim=2] index_array):

    cdef int i, j, number_found = 0
    for i in xrange(rows):
        for j in xrange(cols):
            if inarray[i, j] == search_value:
                index_array[number_found, 0] = i
                index_array[number_found, 1] = j
                number_found = number_found + 1
    
    return number_found

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
@cython.nonecheck(False)
def acs_traceback(np.ndarray[np.float64_t, ndim=1] r_codeword, 
                  trellis, decoding_type,  
                  np.ndarray[np.float64_t, ndim=2] path_metrics, 
                  np.ndarray[np.int_t, ndim=2] paths, 
                  np.ndarray[np.int_t, ndim=2] decoded_symbols,
                  np.ndarray[np.int_t, ndim=1] decoded_bits,
                  int tb_count, int t, int count, 
                  int tb_depth, int current_number_states):

    cdef int state_num, i, j, number_previous_states, previous_state, \
            previous_input, i_codeword, number_found, min_idx, \
            current_state, dec_symbol

    cdef int k = trellis.k
    cdef int n = trellis.n
    cdef int number_states = trellis.number_states
    cdef int number_inputs = trellis.number_inputs

    cdef float branch_metric = 0
    
    cdef np.ndarray[np.int_t, ndim=2] next_state_table = trellis.next_state_table
    cdef np.ndarray[np.int_t, ndim=2] output_table = trellis.output_table
    cdef np.ndarray[np.float64_t, ndim=1] pmetrics = np.empty(number_inputs)
    cdef np.ndarray[np.int_t, ndim=1] i_codeword_array = np.empty(n, 'int')
    cdef np.ndarray[np.int_t, ndim=2] index_array = np.empty([number_states, 2], 'int')
    cdef np.ndarray[np.int_t, ndim=1] decoded_bitarray = np.empty(k, 'int')
    
    # Loop over all the current states (Time instant: t)
    for state_num in xrange(current_number_states):

        # Using the next state table find the previous states and inputs
        # leading into the current state (Trellis)
        number_found = where_c(next_state_table, number_states, number_inputs, state_num, index_array)
           
        # Loop over all the previous states (Time instant: t-1)
        for i in xrange(number_found):
                
            previous_state = index_array[i, 0]
            previous_input = index_array[i, 1]

            # Using the output table, find the ideal codeword 
            i_codeword = output_table[previous_state, previous_input]
            dec2bitarray_c(i_codeword, n, i_codeword_array)
                
            # Compute Branch Metrics
            if decoding_type == 'hard':
                branch_metric = hamming_dist_c(r_codeword, i_codeword_array, n)
            elif decoding_type == 'soft':
                pass
            elif decoding_type == 'unquantized':
                i_codeword_array = 2*i_codeword_array - 1
                branch_metric = euclid_dist_c(r_codeword, i_codeword_array, n)
            else:
                pass

            # ADD operation: Add the branch metric to the 
            # accumulated path metric and store it in the temporary array
            pmetrics[i] = path_metrics[previous_state, 0] + branch_metric

        # COMPARE and SELECT operations
        # Compare and Select the minimum accumulated path metric    
        path_metrics[state_num, 1] = pmetrics.min()
                
        # Store the previous state corresponding to the minimum 
        # accumulated path metric
        min_idx = pmetrics.argmin()
        paths[state_num, tb_count] = index_array[min_idx, 0]
                
        # Store the previous input corresponding to the minimum 
        # accumulated path metric        
        decoded_symbols[state_num, tb_count] = index_array[min_idx, 1]
    
    if t >= tb_depth - 1:
        current_state = path_metrics[:,1].argmin()
        
        # Traceback Loop
        for j in reversed(xrange(1, tb_depth)):

            dec_symbol = decoded_symbols[current_state, j]
            previous_state = paths[current_state, j]
            dec2bitarray_c(dec_symbol, k, decoded_bitarray)
            decoded_bits[(t-tb_depth-1)+(j+1)*k+count:(t-tb_depth-1)+(j+2)*k+count] =  \
                    decoded_bitarray
            current_state = previous_state
        
        paths[:,0:tb_depth-1] = paths[:,1:]
        decoded_symbols[:,0:tb_depth-1] = decoded_symbols[:,1:]
