# Authors: CommPy contributors
# License: BSD 3-Clause

from numpy import array, inf
from numpy.random import randint, randn
from numpy.testing import assert_array_equal, dec, assert_array_less

from commpy.channelcoding.convcode import Trellis, conv_encode, viterbi_decode


class TestConvCode(object):

    @classmethod
    def setup_class(cls):
        cls.trellis = []
        cls.desired_next_state_table = []
        cls.desired_output_table = []

        ### 1/2 - rate codes  ###

        # Convolutional Code 1: G(D) = [1+D^2, 1+D+D^2]
        memory = array([2])
        g_matrix = array([[5, 7]])
        cls.trellis.append(Trellis(memory, g_matrix, code_type='default'))
        cls.desired_next_state_table.append(array([[0, 2],
                                                   [0, 2],
                                                   [1, 3],
                                                   [1, 3]]))
        cls.desired_output_table.append(array([[0, 3],
                                               [3, 0],
                                               [1, 2],
                                               [2, 1]]))

        # Convolutional Code 2: G(D) = [1 1+D+D^2/1+D]
        memory = array([2])
        g_matrix = array([[1, 7]])
        feedback = 5
        cls.trellis.append(Trellis(memory, g_matrix, feedback, 'rsc'))
        cls.desired_next_state_table.append(array([[0, 2],
                                                   [2, 0],
                                                   [1, 3],
                                                   [3, 1]]))
        cls.desired_output_table.append(array([[0, 3],
                                               [0, 3],
                                               [1, 2],
                                               [1, 2]]))

        ### 2/3 - rate codes  ###

        # Convolutional Code 1: G(D) = [[1+D^2, 1+D+D^2 0], [0, D, 1+D]]
        memory = array([2, 1])
        g_matrix = array([[5, 7, 0], [0, 2, 3]])
        cls.trellis.append(Trellis(memory, g_matrix, code_type='default'))
        cls.desired_next_state_table.append(array([[0, 1, 4, 5],
                                                   [0, 1, 4, 5],
                                                   [0, 1, 4, 5],
                                                   [0, 1, 4, 5],
                                                   [2, 3, 6, 7],
                                                   [2, 3, 6, 7],
                                                   [2, 3, 6, 7],
                                                   [2, 3, 6, 7]]))
        cls.desired_output_table.append(array([[0, 1, 6, 7],
                                               [3, 2, 5, 4],
                                               [6, 7, 0, 1],
                                               [5, 4, 3, 2],
                                               [2, 3, 4, 5],
                                               [1, 0, 7, 6],
                                               [4, 5, 2, 3],
                                               [7, 6, 1, 0]]))

        # Convolutional Code 2: G(D) = [[1, 0, 0], [0, 1, 1+D]]; F(D) = [[D, D], [1+D, 1]]
        memory = array([1, 1])
        g_matrix = array([[1, 0, 0], [0, 1, 3]])
        feedback = array([[2, 2], [3, 1]])
        cls.trellis.append(Trellis(memory, g_matrix, feedback, 'rsc'))
        cls.desired_next_state_table.append(array([[0, 1, 1, 0],
                                                   [2, 3, 3, 2],
                                                   [3, 2, 2, 3],
                                                   [1, 0, 0, 1]]))
        cls.desired_output_table.append(array([[0, 3, 4, 7],
                                               [1, 2, 5, 6],
                                               [0, 3, 4, 7],
                                               [1, 2, 5, 6]]))

    @classmethod
    def teardown_class(cls):
        pass

    def test_next_state_table(self):
        for i in range(len(self.trellis)):
            assert_array_equal(self.trellis[i].next_state_table, self.desired_next_state_table[i])

    def test_output_table(self):
        for i in range(len(self.trellis)):
            assert_array_equal(self.trellis[i].output_table, self.desired_output_table[i])

    def test_conv_encode(self):
        pass

    def test_viterbi_decode(self):
        pass

    @dec.slow
    def test_conv_encode_viterbi_decode(self):
        niters = 10
        blocklength = 1000

        for i in range(niters):
            msg = randint(0, 2, blocklength)

            # Previous tests
            for i in range(4):
                coded_bits = conv_encode(msg, self.trellis[i])
                decoded_bits = viterbi_decode(coded_bits.astype(float), self.trellis[i], 15)
                assert_array_equal(decoded_bits[:len(msg)], msg)

                coded_bits = conv_encode(msg, self.trellis[i], termination='cont')
                decoded_bits = viterbi_decode(coded_bits.astype(float), self.trellis[i], 15)
                assert_array_equal(decoded_bits, msg)

                coded_bits = conv_encode(msg, self.trellis[i])
                coded_syms = 2.0 * coded_bits - 1
                decoded_bits = viterbi_decode(coded_syms, self.trellis[i], 15, 'unquantized')
                assert_array_equal(decoded_bits[:len(msg)], msg)

                coded_bits = conv_encode(msg, self.trellis[i])
                coded_syms = 10.0 * coded_bits - 5 + randn(len(coded_bits)) * 2
                decoded_bits = viterbi_decode(coded_syms, self.trellis[i], 15, 'soft')
                assert_array_less((decoded_bits[:len(msg)] - msg).sum(), 0.03 * blocklength)

                coded_bits = conv_encode(msg, self.trellis[i])
                coded_syms = (2.0 * coded_bits - 1) * inf
                decoded_bits = viterbi_decode(coded_syms, self.trellis[i], 15, 'soft')
                assert_array_less((decoded_bits[:len(msg)] - msg).sum(), 0.03 * blocklength)
