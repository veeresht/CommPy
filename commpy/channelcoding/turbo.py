
from numpy import array, append, zeros, exp, pi
from commpy.channelcoding import Trellis, conv_encode, rand_interlv
from commpy.utilities import dec2bitarray, bitarray2dec

def turbo_encode(msg_bits, trellis1, trellis2, interlv_seed):
    
    puncture_matrix = array([[0, 1]])
    sys_stream = append(msg_bits, zeros(trellis1.total_memory, 'int'));
    non_sys_stream_1 = conv_encode(msg_bits, trellis1, puncture_matrix)
    interlv_msg_bits = rand_interlv(msg_bits, interlv_seed)
    non_sys_stream_2 = conv_encode(interlv_msg_bits, trellis2, puncture_matrix)

    return [sys_stream, non_sys_stream_1, non_sys_stream_2]

def map_decode(sys_symbols, non_sys_symbols, trellis, noise_variance):
    
    k = trellis.k
    n = trellis.n
    rate = float(k)/n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs

    msg_length = len(sys_symbols) 

    # Initialize forward state metrics (alpha)
    f_state_metrics = zeros([number_states, 2])
    f_state_metrics[0][0] = 1
    print f_state_metrics
    # Initialize backward state metrics (beta)
    b_state_metrics = zeros(number_states)
    b_state_metrics[number_states-1] = 1 

    # Initialize branch transition probabilities (gamma)
    branch_probs = zeros([number_inputs, number_states, number_states])

    # Forward Recursion
    for time_index in xrange(1, msg_length):
        
        print '======= Time ' + str(time_index) + '============='

        for current_state in xrange(number_states):
            for current_input in xrange(number_inputs):
                next_state = trellis.next_state_table[current_state,current_input]       
                code_symbol = trellis.output_table[current_state][current_input]
                parity_bit = dec2bitarray(code_symbol, n)[1]
                msg_bit = dec2bitarray(code_symbol, n)[0]
                rx_symbol_0 = sys_symbols[time_index-1]
                rx_symbol_1 = non_sys_symbols[time_index-1]
                branch_probs[current_input, current_state, next_state] = \
                _compute_branch_prob(msg_bit, parity_bit, 
                                     rx_symbol_0, rx_symbol_1, 
                                     noise_variance)

                print branch_probs[current_input, current_state, next_state]    
                # Compute the forward state metrics
                f_state_metrics[next_state][1] = \
                        f_state_metrics[next_state][1] + \
                        f_state_metrics[current_state][0] * \
                        branch_probs[current_input, current_state, next_state]
        
        
        # Normalization of the forward state metrics 
        f_state_metrics[:,1] = f_state_metrics[:,1]/f_state_metrics[:,1].sum()
        
        print '======== Branch Probabilities ==========='
        print branch_probs
        print '======== Forward State Metrics =========='
        print f_state_metrics
                
        raw_input()

        f_state_metrics[:,0] = f_state_metrics[:,1]
        f_state_metrics[:,1] = 0

               
def _compute_branch_prob(code_bit_0, code_bit_1, rx_symbol_0, rx_symbol_1, 
                         noise_variance):
    
    code_symbol_0 = 2*code_bit_0 - 1
    code_symbol_1 = 2*code_bit_1 - 1

    return exp( -((rx_symbol_0 - code_symbol_0)*(rx_symbol_0 - code_symbol_0) + 
                  (rx_symbol_1 - code_symbol_1)*(rx_symbol_1 - code_symbol_1)) / 
                  (2*noise_variance) )

    #H = (1/(2*pi*noise_variance))*exp(-((rx_symbol_0+1)*(rx_symbol_0+1) + 
    #                                    (rx_symbol_1+1)*(rx_symbol_1+1)) / 
    #                                    (2*noise_variance))

    #return H*exp(2*(code_bit_0*rx_symbol_0 + code_bit_1*rx_symbol_1)/
    #                noise_variance)  
    

