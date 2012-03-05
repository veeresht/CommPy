
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

""" Turbo Codes """

from numpy import array, append, zeros, exp, pi, log
from commpy.channelcoding import Trellis, conv_encode, rand_interlv, rand_deinterlv
from commpy.utilities import dec2bitarray, bitarray2dec

def turbo_encode(msg_bits, trellis1, trellis2, interlv_seed):
    
    stream = conv_encode(msg_bits, trellis1, 'rsc')
    sys_stream = stream[::2]
    non_sys_stream_1 = stream[1::2]
    
    interlv_msg_bits = rand_interlv(msg_bits, interlv_seed)
    puncture_matrix = array([[0, 1]])
    non_sys_stream_2 = conv_encode(interlv_msg_bits, trellis2, 'rsc', puncture_matrix)

    return [sys_stream, non_sys_stream_1, non_sys_stream_2]

def map_decode(sys_symbols, non_sys_symbols, trellis, noise_variance, L_int):
    
    k = trellis.k
    n = trellis.n
    rate = float(k)/n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs

    msg_length = len(sys_symbols) 

    # Initialize forward state metrics (alpha)
    f_state_metrics = zeros([number_states, 2])
    f_state_metrics[0][0] = 1
    #print f_state_metrics

    # Initialize backward state metrics (beta)
    b_state_metrics = zeros([number_states, msg_length+1])
    b_state_metrics[0][msg_length] = 1 

    # Initialize branch transition probabilities (gamma)
    branch_probs = zeros([number_inputs, number_states, number_states])
    
    app = zeros(number_inputs)
    
    lappr = 0 

    decoded_bits = zeros(msg_length, 'int')
    L_ext = zeros(msg_length)

    # Backward recursion
    for reverse_time_index in reversed(xrange(1, msg_length+1)):
        
        for current_state in xrange(number_states):
            for current_input in xrange(number_inputs):
                next_state = trellis.next_state_table[current_state,current_input]       
                code_symbol = trellis.output_table[current_state][current_input]
                parity_bit = dec2bitarray(code_symbol, n)[1]
                msg_bit = dec2bitarray(code_symbol, n)[0]
                rx_symbol_0 = sys_symbols[reverse_time_index-1]
                rx_symbol_1 = non_sys_symbols[reverse_time_index-1]
                #branch_probs[current_input, current_state, next_state] = \
                branch_prob = _compute_branch_prob(msg_bit, parity_bit, 
                                                   rx_symbol_0, rx_symbol_1, 
                                                   noise_variance)

                b_state_metrics[current_state][reverse_time_index-1] += \
                         (b_state_metrics[next_state][reverse_time_index] *
                           branch_prob)  
        
        b_state_metrics[:,reverse_time_index-1] /= \
            b_state_metrics[:,reverse_time_index-1].sum()
    
    #print "============= Backward State Metrics ================"
    #pprint.pprint(b_state_metrics)

    # Forward Recursion
    for time_index in xrange(1, msg_length+1):
        
        #print '======= Time ' + str(time_index) + '============='
        app[:] = 0
        for current_state in xrange(number_states):
            for current_input in xrange(number_inputs):
                next_state = trellis.next_state_table[current_state,current_input]       
                code_symbol = trellis.output_table[current_state][current_input]
                parity_bit = dec2bitarray(code_symbol, n)[1]
                msg_bit = dec2bitarray(code_symbol, n)[0]
                rx_symbol_0 = sys_symbols[time_index-1]
                rx_symbol_1 = non_sys_symbols[time_index-1]
                #branch_probs[current_input, current_state, next_state] = \
                branch_prob = _compute_branch_prob(msg_bit, parity_bit, 
                                                   rx_symbol_0, rx_symbol_1, 
                                                   noise_variance)

                #print branch_probs[current_input, current_state, next_state]    
                # Compute the forward state metrics
                f_state_metrics[next_state][1] += \
                        f_state_metrics[current_state][0] * branch_prob
                        #branch_probs[current_input, current_state, next_state]                
                        
                # Compute APP
                app[current_input] += (f_state_metrics[current_state][0] * 
                                       branch_prob * 
                                       b_state_metrics[next_state][time_index])

        L_ext[time_index-1] = log(app[1]/app[0])
        lappr = L_int[time_index-1] + L_ext[time_index-1]
        
        #print lappr
        if lappr > 0:
            decoded_bits[time_index-1] = 1
        else:
            decoded_bits[time_index-1] = 0
        
        # Normalization of the forward state metrics 
        f_state_metrics[:,1] = f_state_metrics[:,1]/f_state_metrics[:,1].sum()
        
        #print '======== Branch Probabilities ==========='
        #print branch_probs
        #print '======== Forward State Metrics =========='
        #print f_state_metrics
                
        #raw_input()

        f_state_metrics[:,0] = f_state_metrics[:,1]
        f_state_metrics[:,1] = 0
    
    return [L_ext, decoded_bits]
    #return decoded_bits[0:msg_length-trellis.total_memory]
               
def _compute_branch_prob(code_bit_0, code_bit_1, rx_symbol_0, rx_symbol_1, 
                         noise_variance):
    
    code_symbol_0 = 2*code_bit_0 - 1
    code_symbol_1 = 2*code_bit_1 - 1
    
    # Normalized branch transition probability
    return exp( -((rx_symbol_0 - code_symbol_0)*(rx_symbol_0 - code_symbol_0) + 
                  (rx_symbol_1 - code_symbol_1)*(rx_symbol_1 - code_symbol_1)) / 
                  (2*noise_variance) )


def turbo_decode(sys_symbols, non_sys_symbols_1, non_sys_symbols_2, trellis, 
                 noise_variance, number_iterations, interlv_seed, L_int = None):
    
    if L_int is None:
        L_int = zeros(len(sys_symbols))

    L_int_1 = L_int
    for iteration_count in xrange(number_iterations):
        
        #print "===========" + str(iteration_count) + "=============="

        # MAP Decoder - 1
        [L_ext_1, decoded_bits] = map_decode(sys_symbols, non_sys_symbols_1, 
                                             trellis, noise_variance, L_int_1)
        
        #print " ========= L extrinsic 1 ===================="
        #print L_ext_1

        # Interleave systematic symbols for input to second decoder
        sys_symbols_i = rand_interlv(sys_symbols, interlv_seed)

        L_ext_1 = L_ext_1 - L_int_1
        L_ext_1 = rand_interlv(L_ext_1, interlv_seed)

        # MAP Decoder - 2
        [L_2, decoded_bits] = map_decode(sys_symbols_i, non_sys_symbols_2, 
                                         trellis, noise_variance, L_ext_1)
        
        #print " ============ L extrinsic 2 ======================"
        #print L_2

        L_ext_2 = L_2 - L_ext_1
        L_int_1 = rand_deinterlv(L_ext_2, interlv_seed)
        
    decoded_bits = rand_deinterlv(decoded_bits, interlv_seed)
    return decoded_bits

