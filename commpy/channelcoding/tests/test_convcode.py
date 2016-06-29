

# Authors: Veeresh Taranalli <veeresht@gmail.com>
# License: BSD 3-Clause

from numpy import array
from numpy.random import randint
from numpy.testing import assert_array_equal
from commpy.channelcoding.convcode import Trellis, conv_encode, viterbi_decode

class TestConvCode(object):

    @classmethod
    def setup_class(cls):
        # Convolutional Code 1: G(D) = [1+D^2, 1+D+D^2]
        memory = array([2])
        g_matrix = array([[0o5, 0o7]])
        cls.code_type_1 = 'default'
        cls.trellis_1 = Trellis(memory, g_matrix, 0, cls.code_type_1)
        cls.desired_next_state_table_1 = array([[0, 2],
                                                [0, 2],
                                                [1, 3],
                                                [1, 3]])
        cls.desired_output_table_1 = array([[0, 3],
                                            [3, 0],
                                            [1, 2],
                                            [2, 1]])


        # Convolutional Code 2: G(D) = [1 1+D+D^2/1+D]
        memory = array([2])
        g_matrix = array([[0o1, 0o7]])
        feedback = 0o5
        cls.code_type_2 = 'rsc'
        cls.trellis_2 = Trellis(memory, g_matrix, feedback, cls.code_type_2)
        cls.desired_next_state_table_2 = array([[0, 2],
                                                [2, 0],
                                                [1, 3],
                                                [3, 1]])
        cls.desired_output_table_2 = array([[0, 3],
                                            [0, 3],
                                            [1, 2],
                                            [1, 2]])


    @classmethod
    def teardown_class(cls):
        pass

    def test_next_state_table(self):
        assert_array_equal(self.trellis_1.next_state_table, self.desired_next_state_table_1)
        assert_array_equal(self.trellis_2.next_state_table, self.desired_next_state_table_2)

    def test_output_table(self):
        assert_array_equal(self.trellis_1.output_table, self.desired_output_table_1)
        assert_array_equal(self.trellis_2.output_table, self.desired_output_table_2)

    def test_conv_encode(self):
        pass

    def test_viterbi_decode(self):
        pass

    def test_conv_encode_viterbi_decode(self):
        niters = 10
        blocklength = 1000

        for i in range(niters):
            msg = randint(0, 2, blocklength)

            coded_bits = conv_encode(msg, self.trellis_1)
            decoded_bits = viterbi_decode(coded_bits.astype(float), self.trellis_1, 15)
            assert_array_equal(decoded_bits[:-2], msg)

            coded_bits = conv_encode(msg, self.trellis_1)
            coded_syms = 2.0*coded_bits - 1
            decoded_bits = viterbi_decode(coded_syms, self.trellis_1, 15, 'unquantized')
            assert_array_equal(decoded_bits[:-2], msg)

            coded_bits = conv_encode(msg, self.trellis_2)
            decoded_bits = viterbi_decode(coded_bits.astype(float), self.trellis_2, 15)
            assert_array_equal(decoded_bits[:-2], msg)

            coded_bits = conv_encode(msg, self.trellis_2)
            coded_syms = 2.0*coded_bits - 1
            decoded_bits = viterbi_decode(coded_syms, self.trellis_2, 15, 'unquantized')
            assert_array_equal(decoded_bits[:-2], msg)
