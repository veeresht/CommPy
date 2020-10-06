# Authors: CommPy contributors
# License: BSD 3-Clause

""" Algorithms for Convolutional Codes """

from __future__ import division

import functools
import math
from warnings import warn

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

from commpy.utilities import dec2bitarray, bitarray2dec, hamming_dist, euclid_dist

__all__ = ['Trellis', 'conv_encode', 'viterbi_decode']

class Trellis:
    """
    Class defining a Trellis corresponding to a k/n - rate convolutional code.

    This follow the classical representation. See [1] for instance.

    Input and output are represented as little endian e.g. output = decimal(output[0], output[1] ...).

    Parameters
    ----------
    memory : 1D ndarray of ints
        Number of memory elements per input of the convolutional encoder.
    g_matrix : 2D ndarray of ints (decimal representation)
        Generator matrix G(D) of the convolutional encoder. Each element of G(D) represents a polynomial.
        Coef [i,j] is the influence of input i on output j.
    feedback : 2D ndarray of ints (decimal representation), optional
        Feedback matrix F(D) of the convolutional encoder. Each element of F(D) represents a polynomial.
        Coef [i,j] is the feedback influence of input i on input j.
        *Default* implies no feedback.

        The backwards compatibility version is triggered if feedback is an int.
    code_type : {'default', 'rsc'}, optional
        Use 'rsc' to generate a recursive systematic convolutional code.
        If 'rsc' is specified, then the first 'k x k' sub-matrix of
        G(D) must represent a identity matrix along with a non-zero
        feedback polynomial.
        *Default* is 'default'.
    polynomial_format : {'MSB', 'LSB', 'Matlab'}, optional
        Defines how to interpret g_matrix and feedback. In MSB format, we have 1+D <-> 3 <-> 011.
        In LSB format, which is used in Matlab, we have 1+D <-> 6 <-> 110.
        *Default* is 'MSB' format.

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

    Raises
    ------
    ValueError
        polynomial_format is not 'MSB', 'LSB' or 'Matlab'.

    Examples
    --------
    >>> from numpy import array
    >>> import commpy.channelcoding.convcode as cc
    >>> memory = array([2])
    >>> g_matrix = array([[5, 7]]) # G(D) = [1+D^2, 1+D+D^2]
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
    References
    ----------
    [1] S. Benedetto, R. Garello et G. Montorsi, "A search for good convolutional codes to be used in the
    construction of turbo codes", IEEE Transactions on Communications, vol. 46, n. 9, p. 1101-1005, spet. 1998
    """
    def __init__(self, memory, g_matrix, feedback=None, code_type='default', polynomial_format='MSB'):

        [self.k, self.n] = g_matrix.shape
        self.code_type = code_type
        
        self.total_memory = memory.sum()
        self.number_states = pow(2, self.total_memory)
        self.number_inputs = pow(2, self.k)
        self.next_state_table = np.zeros([self.number_states,
                                          self.number_inputs], 'int')
        self.output_table = np.zeros([self.number_states,
                                      self.number_inputs], 'int')

        if isinstance(feedback, int):
            warn('Trellis  will only accept feedback as a matrix in the future. '
                 'Using the backwards compatibility version that may contain bugs for k > 1 or with LSB format.',
                 DeprecationWarning)

            if code_type == 'rsc':
                for i in range(self.k):
                    g_matrix[i][i] = feedback

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
                                                           memory[l] + 1)

                            # Loop over M delay elements of the shift register
                            # to compute their contribution to the r-th output
                            for i in range(memory[l]):
                                outbits[r] = (outbits[r] + \
                                              (shift_register[i + l] * generator_array[i + 1])) % 2

                            output_generator_array[l] = generator_array[0]
                            if l == 0:
                                feedback_array = (dec2bitarray(feedback, memory[l] + 1)[1:] * shift_register[0:memory[l]]).sum()
                                shift_register[1:memory[l]] = \
                                    shift_register[0:memory[l] - 1]
                                shift_register[0] = (dec2bitarray(current_input,
                                                                  self.k)[0] + feedback_array) % 2
                            else:
                                feedback_array = (dec2bitarray(feedback, memory[l] + 1) *
                                                  shift_register[
                                                  l + memory[l - 1] - 1:l + memory[l - 1] + memory[l] - 1]).sum()
                                shift_register[l + memory[l - 1]:l + memory[l - 1] + memory[l] - 1] = \
                                    shift_register[l + memory[l - 1] - 1:l + memory[l - 1] + memory[l] - 2]
                                shift_register[l + memory[l - 1] - 1] = \
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

        else:
            if polynomial_format == 'MSB':
                bit_order = -1
            elif polynomial_format in ('LSB', 'Matlab'):
                bit_order = 1
            else:
                raise ValueError('polynomial_format must be "LSB", "MSB" or "Matlab"')

            if feedback is None:
                feedback = np.identity(self.k, int)
                if polynomial_format in ('LSB', 'Matlab'):
                    feedback *= 2**memory.max()

            max_values_lign = memory.max() + 1  # Max number of value on a delay lign

            # feedback_array[i] holds the i-th bit corresponding to each feedback polynomial.
            feedback_array = np.zeros((max_values_lign, self.k, self.k), np.int8)
            for i in range(self.k):
                for j in range(self.k):
                    binary_view = dec2bitarray(feedback[i, j], max_values_lign)[::bit_order]
                    feedback_array[:max_values_lign, i, j] = binary_view[-max_values_lign-2:]

            # g_matrix_array[i] holds the i-th bit corresponding to each g_matrix polynomial.
            g_matrix_array = np.zeros((max_values_lign, self.k, self.n), np.int8)
            for i in range(self.k):
                for j in range(self.n):
                    binary_view = dec2bitarray(g_matrix[i, j], max_values_lign)[::bit_order]
                    g_matrix_array[:max_values_lign, i, j] = binary_view[-max_values_lign-2:]

            # shift_regs holds on each column the state of a shift register.
            # The first row is the input of each shift reg.
            shift_regs = np.empty((max_values_lign, self.k), np.int8)

            # Compute the entries in the next state table and the output table
            for current_state in range(self.number_states):
                for current_input in range(self.number_inputs):
                    current_state_array = dec2bitarray(current_state, self.total_memory)

                    # Set the first row as the input.
                    shift_regs[0] = dec2bitarray(current_input, self.k)

                    # Set the other rows based on the current_state
                    idx = 0
                    for idx_mem, mem in enumerate(memory):
                        shift_regs[1:mem+1, idx_mem] = current_state_array[idx:idx + mem]
                        idx += mem

                    # Compute the output table
                    outputs_array = np.einsum('ik,ikl->l', shift_regs, g_matrix_array) % 2
                    self.output_table[current_state, current_input] = bitarray2dec(outputs_array)

                    # Update the first line based on the feedback polynomial
                    np.einsum('ik,ilk->l', shift_regs, feedback_array, out=shift_regs[0])
                    shift_regs %= 2

                    # Update current state array and compute next state table
                    idx = 0
                    for idx_mem, mem in enumerate(memory):
                        current_state_array[idx:idx + mem] = shift_regs[:mem, idx_mem]
                        idx += mem
                    self.next_state_table[current_state, current_input] = bitarray2dec(current_state_array)

    def _generate_grid(self, trellis_length):
        """ Private method """

        grid = np.mgrid[0.12:0.22*trellis_length:(trellis_length+1)*(0+1j),
                        0.1:0.5+self.number_states*0.1:self.number_states*(0+1j)].reshape(2, -1)

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
                        found_index = np.where(self.next_state_table[state_order[state_count_1]] ==
                                                state_order[state_count_2])
                        edge_patch = mpatches.FancyArrow(grid_subset[0,state_count_1]+state_radius,
                                grid_subset[1,state_count_1], dx, dy, width=0.005,
                                length_includes_head = True, color = edge_colors[found_index[0][0]-1])
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
                  state_radius = 0.04, edge_colors = None, save_path = None):
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
        edge_colors : list of hex color codes, optional
            A list of length equal to the number_inputs,
            containing color codes that represent the edge corresponding
            to the input.
        save_path : str or None
            If not None, save the figure to the file specified by its path.
            *Default* is no saving.
        """
        if edge_colors is None:
            edge_colors = [mcolors.hsv_to_rgb((i/self.number_inputs, 1, 1)) for i in range(self.number_inputs)]

        if state_order is None:
            state_order = list(range(self.number_states))

        font = "sans-serif"
        fig = plt.figure(figsize=(12, 6), dpi=150)
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
        plt.legend(edge_patches, [str(i) + "-input" for i in range(self.number_inputs)])
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)

    def visualize_fsm(self, state_order=None, state_radius=0.04, edge_colors=None, save_path=None):
        """ Plot the FSM corresponding to the the trellis

        This method is not intended to display large FSMs and its use is advisable only for simple trellises.

        Parameters
        ----------
        state_order : list of ints, optional
            Specifies the order in the which the states of the trellis are to be displayed starting from the top in the
            plot.
            *Default* order is [0,...,number_states-1]
        state_radius : float, optional
            Radius of each state (circle) in the plot.
            *Default* value is 0.04
        edge_colors : list of hex color codes, optional
            A list of length equal to the number_inputs, containing color codes that represent the edge corresponding to
            the input.
        save_path : str or None
            If not None, save the figure to the file specified by its path.
            *Default* is no saving.
        """
        # Default arguments
        if edge_colors is None:
            edge_colors = [mcolors.hsv_to_rgb((i/self.number_inputs, 1, 1)) for i in range(self.number_inputs)]

        if state_order is None:
            state_order = list(range(self.number_states))

        # Init the figure
        ax = plt.axes((0, 0, 1, 1))

        # Plot states
        radius = state_radius * self.number_states
        angles = 2 * np.pi / self.number_states * np.arange(self.number_states)
        positions = [(radius * math.cos(angle), radius * math.sin(angle)) for angle in angles]

        state_patches = []
        arrows = []
        for idx, state in enumerate(state_order):
            state_patches.append(mpatches.Circle(positions[idx], state_radius, color="#003399", ec="#cccccc"))
            plt.text(positions[idx][0], positions[idx][1], str(state), ha='center', va='center', size=20)

            # Plot transition
            for input in range(self.number_inputs):
                next_state = self.next_state_table[state, input]
                next_idx = (state_order == next_state).nonzero()[0][0]
                output = self.output_table[state, input]

                # Transition arrow
                if next_state == state:
                    # Positions
                    arrow_start_x = positions[idx][0] + state_radius * math.cos(angles[idx] + math.pi / 6)
                    arrow_start_y = positions[idx][1] + state_radius * math.sin(angles[idx] + math.pi / 6)
                    arrow_end_x = positions[idx][0] + state_radius * math.cos(angles[idx] - math.pi / 6)
                    arrow_end_y = positions[idx][1] + state_radius * math.sin(angles[idx] - math.pi / 6)
                    arrow_mid_x = positions[idx][0] + state_radius * 2 * math.cos(angles[idx])
                    arrow_mid_y = positions[idx][1] + state_radius * 2 * math.sin(angles[idx])

                    # Add text
                    plt.text(arrow_mid_x, arrow_mid_y, '({})'.format(output),
                             ha='center', va='center', backgroundcolor=edge_colors[input])

                else:
                    # Positions
                    dx = positions[next_idx][0] - positions[idx][0]
                    dy = positions[next_idx][1] - positions[idx][1]
                    relative_angle = math.atan(dy / dx) + np.where(dx > 0, 0, math.pi)

                    arrow_start_x = positions[idx][0] + state_radius * math.cos(relative_angle + math.pi * 0.05)
                    arrow_start_y = positions[idx][1] + state_radius * math.sin(relative_angle + math.pi * 0.05)
                    arrow_end_x = positions[next_idx][0] - state_radius * math.cos(relative_angle - math.pi * 0.05)
                    arrow_end_y = positions[next_idx][1] - state_radius * math.sin(relative_angle - math.pi * 0.05)
                    arrow_mid_x = (arrow_start_x + arrow_end_x) / 2 + \
                                   radius * 0.25 * math.cos((angles[idx] + angles[next_idx]) / 2) * np.sign(dx)
                    arrow_mid_y = (arrow_start_y + arrow_end_y) / 2 + \
                                   radius * 0.25 * math.sin((angles[idx] + angles[next_idx]) / 2) * np.sign(dx)
                    text_x = arrow_mid_x + 0.01 * math.cos((angles[idx] + angles[next_idx]) / 2)
                    text_y = arrow_mid_y + 0.01 * math.sin((angles[idx] + angles[next_idx]) / 2)

                    # Add text
                    plt.text(text_x, text_y, '({})'.format(output),
                             ha='center', va='center', backgroundcolor=edge_colors[input])

                # Path creation
                codes = (mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3)
                verts = ((arrow_start_x, arrow_start_y),
                         (arrow_mid_x, arrow_mid_y),
                         (arrow_end_x, arrow_end_y))
                path = mpath.Path(verts, codes)

                # Plot arrow
                arrow = mpatches.FancyArrowPatch(path=path, mutation_scale=20, color=edge_colors[input])
                ax.add_artist(arrow)
                arrows.append(arrow)

        # Format and plot
        ax.set_xlim(radius * -2, radius * 2)
        ax.set_ylim(radius * -2, radius * 2)
        ax.add_collection(PatchCollection(state_patches, True))
        plt.legend(arrows, [str(i) + "-input" for i in range(self.number_inputs)], loc='lower right')
        plt.text(0, 1.5 * radius, 'Finite State Machine (output on transition)', ha='center', size=18)
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)


def conv_encode(message_bits, trellis, termination = 'term', puncture_matrix=None):
    """
    Encode bits using a convolutional code.
    Parameters
    ----------
    message_bits : 1D ndarray containing {0, 1}
        Stream of bits to be convolutionally encoded.
    trellis: pre-initialized Trellis structure.
    termination: {'cont', 'term'}, optional
        Create ('term') or not ('cont') termination bits.
    puncture_matrix: 2D ndarray containing {0, 1}, optional
        Matrix used for the puncturing algorithm
    Returns
    -------
    coded_bits : 1D ndarray containing {0, 1}
        Encoded bit stream.
    """

    k = trellis.k
    n = trellis.n
    total_memory = trellis.total_memory
    rate = float(k)/n
    
    code_type = trellis.code_type

    if puncture_matrix is None:
        puncture_matrix = np.ones((trellis.k, trellis.n))

    number_message_bits = np.size(message_bits)
    
    if termination == 'cont':
        inbits = message_bits
        number_inbits = number_message_bits
        number_outbits = int(number_inbits/rate)
    else:
        # Initialize an array to contain the message bits plus the truncation zeros
        if code_type == 'rsc':
            inbits = message_bits
            number_inbits = number_message_bits
            number_outbits = int((number_inbits + k * total_memory)/rate)
        else:
            number_inbits = number_message_bits + total_memory + total_memory % k
            inbits = np.zeros(number_inbits, 'int')
            # Pad the input bits with M zeros (L-th terminated truncation)
            inbits[0:number_message_bits] = message_bits
            number_outbits = int(number_inbits/rate)

    outbits = np.zeros(number_outbits, 'int')
    if puncture_matrix is not None:
        p_outbits = np.zeros(number_outbits, 'int')
    else:
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

    if code_type == 'rsc' and termination == 'term':
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

    number_found = 0
    res = np.where(inarray == search_value)
    i_s, j_s = res
    for i, j in zip(i_s, j_s):
        if inarray[i, j] == search_value:
            index_array[number_found, 0] = i
            index_array[number_found, 1] = j
            number_found += 1

    return number_found


@functools.lru_cache(maxsize=128, typed=False)
def _compute_branch_metrics(decoding_type, _r_codeword: tuple, _i_codeword_array: tuple):
    r_codeword = np.array(_r_codeword)
    i_codeword_array = np.array(_i_codeword_array)
    if decoding_type == 'hard':
        return hamming_dist(r_codeword.astype(int), i_codeword_array.astype(int))
    elif decoding_type == 'soft':
        neg_LL_0 = np.log(np.exp(r_codeword) + 1)  # negative log-likelihood to have received a 0
        neg_LL_1 = neg_LL_0 - r_codeword  # negative log-likelihood to have received a 1
        return np.where(i_codeword_array, neg_LL_1, neg_LL_0).sum()
    elif decoding_type == 'unquantized':
        i_codeword_array = 2 * i_codeword_array - 1
        return euclid_dist(r_codeword, i_codeword_array)


def _acs_traceback(r_codeword, trellis, decoding_type,
                   path_metrics, paths, decoded_symbols,
                   decoded_bits, tb_count, t, count,
                   tb_depth, current_number_states):

    k = trellis.k
    n = trellis.n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs

    branch_metric = 0.0

    next_state_table = trellis.next_state_table
    output_table = trellis.output_table
    pmetrics = np.empty(number_inputs)
    index_array = np.empty([number_states, 2], 'int')

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
            i_codeword_array = dec2bitarray(i_codeword, n)

            # Compute Branch Metrics
            branch_metric = _compute_branch_metrics(decoding_type, tuple(r_codeword), tuple(i_codeword_array))

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
            decoded_bits[t - tb_depth + 1 + (j - 1) * k + count:t - tb_depth + 1 + j * k + count] = decoded_bitarray
            current_state = previous_state

        paths[:,0:tb_depth-1] = paths[:,1:]
        decoded_symbols[:,0:tb_depth-1] = decoded_symbols[:,1:]



def viterbi_decode(coded_bits, trellis, tb_depth=None, decoding_type='hard'):
    """
    Decodes a stream of convolutionally encoded bits using the Viterbi Algorithm.
    Parameters
    ----------
    coded_bits : 1D ndarray
        Stream of convolutionally encoded bits which are to be decoded.
    treillis : treillis object
        Treillis representing the convolutional code.
    tb_depth : int
        Traceback depth.
        *Default* is 5 times the number of memories in the code.
    decoding_type : str {'hard', 'soft', 'unquantized'}
        The type of decoding to be used.
        'hard' option is used for hard inputs (bits) to the decoder, e.g., BSC channel.
        'soft' option is used for soft inputs (LLRs) to the decoder. LLRs are clipped in [-500, 500].
        'unquantized' option is used for soft inputs (real numbers) to the decoder, e.g., BAWGN channel.
    Returns
    -------
    decoded_bits : 1D ndarray
        Decoded bit stream.
    Raises
    ------
    ValueError
                If decoding_type is something else than 'hard', 'soft' or 'unquantized'.
    References
    ----------
    .. [1] Todd K. Moon. Error Correction Coding: Mathematical Methods and
        Algorithms. John Wiley and Sons, 2005.
    """

    # k = Rows in G(D), n = columns in G(D)
    k = trellis.k
    n = trellis.n
    rate = k/n
    total_memory = trellis.total_memory

    # Number of message bits after decoding
    L = int(len(coded_bits)*rate)

    if tb_depth is None:
        tb_depth = min(5 * total_memory, L)


    path_metrics = np.full((trellis.number_states, 2), np.inf)
    path_metrics[0][0] = 0
    paths = np.empty((trellis.number_states, tb_depth), 'int')
    paths[0][0] = 0

    decoded_symbols = np.zeros([trellis.number_states, tb_depth], 'int')
    decoded_bits = np.empty(int(math.ceil((L + tb_depth) / k) * k), 'int')
    r_codeword = np.zeros(n, 'int')

    tb_count = 1
    count = 0
    current_number_states = trellis.number_states

    if decoding_type == 'soft':
        coded_bits = coded_bits.clip(-500, 500)

    for t in range(1, int((L+total_memory)/k)):
        # Get the received codeword corresponding to t
        if t <= L // k:
            r_codeword = coded_bits[(t-1)*n:t*n]
        # Pad with '0'
        else:
            if decoding_type == 'hard':
                r_codeword[:] = 0
            elif decoding_type == 'soft':
                r_codeword[:] = 0
            elif decoding_type == 'unquantized':
                r_codeword[:] = -1
            else:
                raise ValueError('The available decoding types are "hard", "soft" and "unquantized')

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

    return decoded_bits[:L]


def puncturing(message: np.ndarray, punct_vec: np.ndarray) -> np.ndarray:
    """
    Applying of the punctured procedure.
    Parameters
    ----------
    message : 1D ndarray
        Input message {0,1} bit array.
    punct_vec : 1D ndarray
        Puncturing vector {0,1} bit array.
    Returns
    -------
    punctured : 1D ndarray
        Output punctured vector {0,1} bit array.
    """
    shift = 0
    N = len(punct_vec)
    punctured = []
    for idx, item in enumerate(message):
        if punct_vec[idx-shift*N] == 1:
            punctured.append(item)
        if idx%N == 0:
            shift = shift + 1
    return np.array(punctured)


def depuncturing(punctured: np.ndarray, punct_vec: np.ndarray, shouldbe: int) -> np.ndarray:
    """
    Applying of the inserting zeros procedure.
    Parameters
    ----------
    punctured : 1D ndarray
        Input punctured message {0,1} bit array.
    punct_vec : 1D ndarray
        Puncturing vector {0,1} bit array.
    shouldbe : int 
        Length of the initial message (before puncturing).
    Returns
    -------
    depunctured : 1D ndarray
        Output vector {0,1} bit array.
    """
    shift = 0
    shift2 = 0
    N = len(punct_vec)
    depunctured = np.zeros((shouldbe,))
    for idx, item in enumerate(depunctured):
        if punct_vec[idx - shift*N] == 1:
            depunctured[idx] = float(punctured[idx-shift2])
        else:
            shift2 = shift2 + 1
        if idx%N == 0:
            shift = shift + 1
    return depunctured
