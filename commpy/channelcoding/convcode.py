

# Authors: Veeresh Taranalli <veeresht@gmail.com>
# License: BSD 3-Clause

""" Algorithms for Convolutional Codes """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

from commpy.utilities import dec2bitarray, bitarray2dec, hamming_dist, euclid_dist
#from commpy.channelcoding.acstb import acs_traceback

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

        If 'rsc' is specified, then the first 'k x k' sub-matrix of

        G(D) must represent a identity matrix along with a non-zero
        feedback polynomial.


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

    number_inputs : int
        Number of branches from each state in the convolutional code trellis.

    next_state_table : 2D ndarray of ints
        Table representing the state transition matrix of the
        convolutional code trellis. Rows represent current states and
        columns represent current inputs in decimal. Elements represent the
        corresponding next states in decimal.

    output_table : 2D ndarray of ints
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
    >>> print trellis.number_inputs
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
            for i in range(self.k):
                g_matrix[i][i] = feedback

        self.total_memory = memory.sum()
        self.number_states = pow(2, self.total_memory)
        self.number_inputs = pow(2, self.k)
        self.next_state_table = np.zeros([self.number_states,
                                          self.number_inputs], 'int')
        self.output_table = np.zeros([self.number_states,
                                      self.number_inputs], 'int')

        # Compute the entries in the next state table and the output table
        for current_state in range(self.number_states):

            for current_input in range(self.number_inputs):
                outbits = np.zeros(self.n, 'int')

                # Compute the values in the output_table
                for r in range(self.n):

                    output_generator_array = np.zeros(self.k, 'int')
                    shift_register = dec2bitarray(current_state,
                                                  self.total_memory)

                    for l in range(self.k):

                        # Convert the number representing a polynomial into a
                        # bit array
                        generator_array = dec2bitarray(g_matrix[l][r],
                                                       memory[l]+1)

                        # Loop over M delay elements of the shift register
                        # to compute their contribution to the r-th output
                        for i in range(memory[l]):
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


    def _generate_grid(self, trellis_length):
        """ Private method """

        grid = np.mgrid[0.12:0.22*trellis_length:(trellis_length+1)*(0+1j),
                        0.1:0.1+self.number_states*0.1:self.number_states*(0+1j)].reshape(2, -1)

        return grid

    def _generate_states(self, trellis_length, grid, state_order, state_radius, font):
        """ Private method """
        state_patches = []

        for state_count in range(self.number_states * trellis_length):
            state_patch = mpatches.Circle(grid[:,state_count], state_radius,
                    color="#003399", ec="#cccccc")
            state_patches.append(state_patch)
            plt.text(grid[0, state_count], grid[1, state_count]-0.02,
                    str(state_order[state_count % self.number_states]),
                    ha="center", family=font, size=20, color="#ffffff")

        return state_patches

    def _generate_edges(self, trellis_length, grid, state_order, state_radius, edge_colors):
        """ Private method """
        edge_patches = []

        for current_time_index in range(trellis_length-1):
            grid_subset = grid[:,self.number_states * current_time_index:]
            for state_count_1 in range(self.number_states):
                input_count = 0
                for state_count_2 in range(self.number_states):
                    dx = grid_subset[0, state_count_2+self.number_states] - grid_subset[0,state_count_1] - 2*state_radius
                    dy = grid_subset[1, state_count_2+self.number_states] - grid_subset[1,state_count_1]
                    if np.count_nonzero(self.next_state_table[state_order[state_count_1],:] == state_order[state_count_2]):
                        found_index = np.where(self.next_state_table[state_order[state_count_1],:] ==
                                                state_order[state_count_2])
                        edge_patch = mpatches.FancyArrow(grid_subset[0,state_count_1]+state_radius,
                                grid_subset[1,state_count_1], dx, dy, width=0.005,
                                length_includes_head = True, color = edge_colors[found_index[0][0]])
                        edge_patches.append(edge_patch)
                        input_count = input_count + 1

        return edge_patches

    def _generate_labels(self, grid, state_order, state_radius, font):
        """ Private method """

        for state_count in range(self.number_states):
            for input_count in range(self.number_inputs):
                edge_label = str(input_count) + "/" + str(
                        self.output_table[state_order[state_count], input_count])
                plt.text(grid[0, state_count]-1.5*state_radius,
                         grid[1, state_count]+state_radius*(1-input_count-0.7),
                         edge_label, ha="center", family=font, size=14)


    def visualize(self, trellis_length = 2, state_order = None,
                  state_radius = 0.04, edge_colors = None):
        """ Plot the trellis diagram.

        Parameters
        ----------
        trellis_length : int, optional
            Specifies the number of time steps in the trellis diagram.
            Default value is 2.

        state_order : list of ints, optional
            Specifies the order in the which the states of the trellis
            are to be displayed starting from the top in the plot.
            Default order is [0,...,number_states-1]

        state_radius : float, optional
            Radius of each state (circle) in the plot.
            Default value is 0.04

        edge_colors = list of hex color codes, optional
            A list of length equal to the number_inputs,
            containing color codes that represent the edge corresponding
            to the input.

        """
        if edge_colors is None:
            edge_colors = ["#9E1BE0", "#06D65D"]

        if state_order is None:
            state_order = range(self.number_states)

        font = "sans-serif"
        fig = plt.figure()
        ax = plt.axes([0,0,1,1])
        trellis_patches = []

        state_order.reverse()

        trellis_grid = self._generate_grid(trellis_length)
        state_patches = self._generate_states(trellis_length, trellis_grid,
                                              state_order, state_radius, font)
        edge_patches = self._generate_edges(trellis_length, trellis_grid,
                                            state_order, state_radius,
                                            edge_colors)
        self._generate_labels(trellis_grid, state_order, state_radius, font)

        trellis_patches.extend(state_patches)
        trellis_patches.extend(edge_patches)

        collection = PatchCollection(trellis_patches, match_original=True)
        ax.add_collection(collection)
        ax.set_xticks([])
        ax.set_yticks([])
        #plt.legend([edge_patches[0], edge_patches[1]], ["1-input", "0-input"])
        plt.show()


def conv_encode(message_bits, trellis, code_type = 'default', puncture_matrix=None):
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

    if puncture_matrix is None:
        puncture_matrix = np.ones((trellis.k, trellis.n))

    number_message_bits = np.size(message_bits)

    # Initialize an array to contain the message bits plus the truncation zeros
    if code_type == 'default':
        inbits = np.zeros(number_message_bits + total_memory + total_memory % k,
                      'int')
        number_inbits = number_message_bits + total_memory + total_memory % k

        # Pad the input bits with M zeros (L-th terminated truncation)
        inbits[0:number_message_bits] = message_bits
        number_outbits = int(number_inbits/rate)

    else:
        inbits = message_bits
        number_inbits = number_message_bits
        number_outbits = int((number_inbits + total_memory)/rate)

    outbits = np.zeros(number_outbits, 'int')
    p_outbits = np.zeros(int(number_outbits*
            puncture_matrix[0:].sum()/np.size(puncture_matrix, 1)), 'int')
    next_state_table = trellis.next_state_table
    output_table = trellis.output_table

    # Encoding process - Each iteration of the loop represents one clock cycle
    current_state = 0
    j = 0

    for i in range(int(number_inbits/k)): # Loop through all input bits
        current_input = bitarray2dec(inbits[i*k:(i+1)*k])
        current_output = output_table[current_state][current_input]
        outbits[j*n:(j+1)*n] = dec2bitarray(current_output, n)
        current_state = next_state_table[current_state][current_input]
        j += 1

    if code_type == 'rsc':

        term_bits = dec2bitarray(current_state, trellis.total_memory)
        term_bits = term_bits[::-1]
        for i in range(trellis.total_memory):
            current_input = bitarray2dec(term_bits[i*k:(i+1)*k])
            current_output = output_table[current_state][current_input]
            outbits[j*n:(j+1)*n] = dec2bitarray(current_output, n)
            current_state = next_state_table[current_state][current_input]
            j += 1

    j = 0
    for i in range(number_outbits):
        if puncture_matrix[0][i % np.size(puncture_matrix, 1)] == 1:
            p_outbits[j] = outbits[i]
            j = j + 1

    return p_outbits


def _where_c(inarray, rows, cols, search_value, index_array):

    #cdef int i, j,
    number_found = 0
    for i in range(rows):
        for j in range(cols):
            if inarray[i, j] == search_value:
                index_array[number_found, 0] = i
                index_array[number_found, 1] = j
                number_found += 1

    return number_found


def _acs_traceback(r_codeword, trellis, decoding_type,
                   path_metrics, paths, decoded_symbols,
                   decoded_bits, tb_count, t, count,
                   tb_depth, current_number_states):

    #cdef int state_num, i, j, number_previous_states, previous_state, \
    #        previous_input, i_codeword, number_found, min_idx, \
    #        current_state, dec_symbol

    k = trellis.k
    n = trellis.n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs

    branch_metric = 0.0

    next_state_table = trellis.next_state_table
    output_table = trellis.output_table
    pmetrics = np.empty(number_inputs)
    i_codeword_array = np.empty(n, 'int')
    index_array = np.empty([number_states, 2], 'int')
    decoded_bitarray = np.empty(k, 'int')

    # Loop over all the current states (Time instant: t)
    for state_num in range(current_number_states):

        # Using the next state table find the previous states and inputs
        # leading into the current state (Trellis)
        number_found = _where_c(next_state_table, number_states, number_inputs, state_num, index_array)

        # Loop over all the previous states (Time instant: t-1)
        for i in range(number_found):

            previous_state = index_array[i, 0]
            previous_input = index_array[i, 1]

            # Using the output table, find the ideal codeword
            i_codeword = output_table[previous_state, previous_input]
            #dec2bitarray_c(i_codeword, n, i_codeword_array)
            i_codeword_array = dec2bitarray(i_codeword, n)

            # Compute Branch Metrics
            if decoding_type == 'hard':
                #branch_metric = hamming_dist_c(r_codeword.astype(int), i_codeword_array.astype(int), n)
                branch_metric = hamming_dist(r_codeword.astype(int), i_codeword_array.astype(int))
            elif decoding_type == 'soft':
                pass
            elif decoding_type == 'unquantized':
                i_codeword_array = 2*i_codeword_array - 1
                branch_metric = euclid_dist(r_codeword, i_codeword_array)
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
        for j in reversed(range(1, tb_depth)):

            dec_symbol = decoded_symbols[current_state, j]
            previous_state = paths[current_state, j]
            decoded_bitarray = dec2bitarray(dec_symbol, k)
            decoded_bits[(t-tb_depth-1)+(j+1)*k+count:(t-tb_depth-1)+(j+2)*k+count] =  \
                    decoded_bitarray
            current_state = previous_state

        paths[:,0:tb_depth-1] = paths[:,1:]
        decoded_symbols[:,0:tb_depth-1] = decoded_symbols[:,1:]



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

    decoding_type : str {'hard', 'unquantized'}
        The type of decoding to be used.
        'hard' option is used for hard inputs (bits) to the decoder, e.g., BSC channel.
        'unquantized' option is used for soft inputs (real numbers) to the decoder, e.g., BAWGN channel.

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
    path_metrics[:, :] = 1000000
    path_metrics[0][0] = 0
    paths = np.empty([number_states, tb_depth], 'int')
    paths[:, :] = 1000000
    paths[0][0] = 0

    decoded_symbols = np.zeros([number_states, tb_depth], 'int')
    decoded_bits = np.zeros(L+tb_depth+k, 'int')
    r_codeword = np.zeros(n, 'int')

    tb_count = 1
    count = 0
    current_number_states = number_states

    for t in range(1, int((L+total_memory+total_memory%k)/k) + 1):
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
                r_codeword = 2*r_codeword - 1
            else:
                pass

        _acs_traceback(r_codeword, trellis, decoding_type, path_metrics, paths,
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

    return decoded_bits[0:len(decoded_bits)-tb_depth-1]
