
import numpy as np

cimport numpy as np
cimport cython

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
@cython.nonecheck(False)
def backward_recursion(trellis, int msg_length, float noise_variance,
                       np.ndarray[np.float64_t, ndim=1] sys_symbols,
                       np.ndarray[np.float64_t, ndim=1] non_sys_symbols,
                       np.ndarray[np.float64_t, ndim=3] branch_probs,
                       np.ndarray[np.float64_t, ndim=2] priors,
                       np.ndarray[np.float64_t, ndim=2] b_state_metrics):

    cdef int reverse_time_index, current_state, current_input, next_state, \
             code_symbol, parity_bit, msg_bit, n, number_states, number_inputs
    
    cdef np.float64_t rx_symbol_0, rx_symbol_1, branch_prob, L_k
 
    n = trellis.n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs

    cdef np.ndarray[np.int_t, ndim=1] codeword_array = np.empty(n, 'int')
    cdef np.ndarray[np.int_t, ndim=2] next_state_table = trellis.next_state_table
    cdef np.ndarray[np.int_t, ndim=2] output_table = trellis.output_table
    #cdef np.ndarray[np.float64_t, ndim=1] priors = np.empty(number_inputs)

    # Backward recursion
    for reverse_time_index in reversed(xrange(1, msg_length+1)):
        
        #p_0 = 1/(1+np.exp(L_int[reverse_time_index-1]))
        #p_1 = np.exp(L_int[reverse_time_index-1])/(1+np.exp(L_int[reverse_time_index-1]))

        #L_k = L_int[reverse_time_index-1]
        #for current_input in xrange(number_inputs):
        #    priors[current_input] = np.exp(current_input*L_k)/(1 + np.exp(L_k))

        #priors = priors * 100
        #priors = priors/(priors.sum())
        
        #print "Time Index" + str(reverse_time_index) + ":",
        #print priors, L_k
        for current_state in xrange(number_states):
            for current_input in xrange(number_inputs):
                next_state = next_state_table[current_state, current_input]       
                code_symbol = output_table[current_state, current_input]
                dec2bitarray_c(code_symbol, n, codeword_array)
                parity_bit = codeword_array[1]
                msg_bit = codeword_array[0]
                rx_symbol_0 = sys_symbols[reverse_time_index-1]
                rx_symbol_1 = non_sys_symbols[reverse_time_index-1]
                branch_prob = compute_branch_prob_c(msg_bit, parity_bit, rx_symbol_0, rx_symbol_1, noise_variance)
                branch_probs[current_input, current_state, reverse_time_index-1] = branch_prob
                b_state_metrics[current_state, reverse_time_index-1] += \
                        (b_state_metrics[next_state, reverse_time_index] * branch_prob * 
                                priors[current_input, reverse_time_index-1]) #/(3.14159*noise_variance)))  
        
        b_state_metrics[:,reverse_time_index-1] /= \
            b_state_metrics[:,reverse_time_index-1].sum()

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
@cython.nonecheck(False)
def forward_recursion_decoding(trellis, mode, int msg_length, float noise_variance,
                               np.ndarray[np.float64_t, ndim=1] sys_symbols,
                               np.ndarray[np.float64_t, ndim=1] non_sys_symbols,
                               np.ndarray[np.float64_t, ndim=2] b_state_metrics,
                               np.ndarray[np.float64_t, ndim=2] f_state_metrics,
                               np.ndarray[np.float64_t, ndim=3] branch_probs,
                               np.ndarray[np.float64_t, ndim=1] app,
                               np.ndarray[np.float64_t, ndim=1] L_int,
                               np.ndarray[np.float64_t, ndim=2] priors,
                               np.ndarray[np.float64_t, ndim=1] L_ext,
                               np.ndarray[np.int_t, ndim=1] decoded_bits):
    
    cdef int time_index, current_state, current_input, next_state, \
             code_symbol, parity_bit, msg_bit, n, number_states, number_inputs
    
    cdef float rx_symbol_0, rx_symbol_1, branch_prob, lappr, L_k
 
    n = trellis.n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs

    cdef np.ndarray[np.int_t, ndim=1] codeword_array = np.empty(n, 'int')
    cdef np.ndarray[np.int_t, ndim=2] next_state_table = trellis.next_state_table
    cdef np.ndarray[np.int_t, ndim=2] output_table = trellis.output_table
    #cdef np.ndarray[np.float64_t, ndim=1] priors = np.empty(number_inputs)

    # Forward Recursion
    for time_index in xrange(1, msg_length+1):
        
        #L_k = L_int[time_index-1]
        #for current_input in xrange(number_inputs):
        #    priors[current_input] = np.exp(current_input*L_k)/(1 + np.exp(L_k))

        #print '======= Time ' + str(time_index) + '============='
        app[:] = 0
        for current_state in xrange(number_states):
            for current_input in xrange(number_inputs):
                next_state = next_state_table[current_state, current_input]       
                #code_symbol = output_table[current_state, current_input]
                #dec2bitarray_c(code_symbol, n, codeword_array)
                #parity_bit = codeword_array[1]
                #msg_bit = codeword_array[0]
                #rx_symbol_0 = sys_symbols[time_index-1]
                #rx_symbol_1 = non_sys_symbols[time_index-1]
                #branch_prob = compute_branch_prob_c(msg_bit, parity_bit, rx_symbol_0, rx_symbol_1, noise_variance)
                branch_prob = branch_probs[current_input, current_state, time_index-1]
                # Compute the forward state metrics
                f_state_metrics[next_state, 1] += f_state_metrics[current_state, 0] * branch_prob * \
                         priors[current_input, time_index-1] #/(3.14159*noise_variance))    
                        
                # Compute APP
                app[current_input] += \
                        f_state_metrics[current_state, 0] *\
                        branch_prob * \
                        b_state_metrics[next_state, time_index]
        
        #print '======== Forward State Metrics =========='
        #print f_state_metrics
        #print '======== Branch Probabilities ==========='
        #print branch_probs

        #print app[0], app[1]
        #print np.log(app[1]/app[0]), 
        #L_ext[time_index-1] = np.log(app[1]/app[0])
        lappr = L_int[time_index-1] + np.log(app[1]/app[0])
        L_ext[time_index-1] = lappr
        
        #print lappr
        if mode == 'decode':
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
        #print '======== Backward State Metrics ========='
        #print b_state_metrics
                
        #raw_input()

        f_state_metrics[:,0] = f_state_metrics[:,1]
        f_state_metrics[:,1] = 0.0


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
@cython.nonecheck(False)
cdef float compute_branch_prob_c(float code_bit_0, float code_bit_1, 
                                 float rx_symbol_0, float rx_symbol_1, 
                                 float noise_variance):
    
    cdef np.float64_t code_symbol_0, code_symbol_1, branch_prob, x, y

    code_symbol_0 = 2*code_bit_0 - 1
    code_symbol_1 = 2*code_bit_1 - 1
    
    x = rx_symbol_0 - code_symbol_0
    y = rx_symbol_1 - code_symbol_1

    # Normalized branch transition probability
    branch_prob = np.exp(-(x*x + y*y)/(2*noise_variance))
    #print branch_prob
    #print x, y, noise_variance
    #raw_input()
    return branch_prob

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void dec2bitarray_c(int in_number, int bit_width, np.ndarray[np.int_t, ndim=1] bitarray):
    
    cdef int i
    
    for i in xrange(bit_width):
        bitarray[bit_width-i-1] = (in_number & 1)
        in_number = in_number >> 1


