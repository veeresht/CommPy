# Authors: CommPy contributors
# License: BSD 3-Clause

import os
from tempfile import TemporaryDirectory

from nose.plugins.attrib import attr
from numpy import array, sqrt, zeros, zeros_like
from numpy.random import randn, choice
from numpy.testing import assert_allclose, assert_equal, assert_raises

from commpy.channelcoding.ldpc import get_ldpc_code_params, ldpc_bp_decode, write_ldpc_params, \
    triang_ldpc_systematic_encode
from commpy.utilities import hamming_dist


class TestLDPCCode(object):

    @classmethod
    def setup_class(cls):
        cls.dir = os.path.dirname(__file__)

    @classmethod
    def teardown_class(cls):
        pass

    @attr('slow')
    def test_ldpc_bp_decode(self):
        ldpc_design_file = os.path.join(self.dir, '../designs/ldpc/gallager/96.33.964.txt')
        ldpc_code_params = get_ldpc_code_params(ldpc_design_file)

        for n_blocks in (1, 2):
            N = 96 * n_blocks
            rate = 0.5
            Es = 1.0
            snr_list = array([2.0, 2.5])
            niters = 10000000
            tx_codeword = zeros(N, int)
            ldpcbp_iters = 100

            for decoder_algorithm in ('MSA', 'SPA'):
                fer_array_ref = array((.2, .1))
                fer_array_test = zeros(len(snr_list))

                for idx, ebno in enumerate(snr_list):

                    noise_std = 1/sqrt((10**(ebno/10.0))*rate*2/Es)
                    fer_cnt_bp = 0

                    for iter_cnt in range(niters):

                        awgn_array = noise_std * randn(N)
                        rx_word = 1-(2*tx_codeword) + awgn_array
                        rx_llrs = 2.0*rx_word/(noise_std**2)

                        [dec_word, _] = ldpc_bp_decode(rx_llrs, ldpc_code_params, decoder_algorithm, ldpcbp_iters)

                        if hamming_dist(tx_codeword, dec_word.reshape(-1)):
                            fer_cnt_bp += 1

                        if fer_cnt_bp >= 50:
                            fer_array_test[idx] = float(fer_cnt_bp) / (iter_cnt + 1) / n_blocks
                            break

                assert_allclose(fer_array_test, fer_array_ref, rtol=.6, atol=0,
                                err_msg=decoder_algorithm + ' algorithm does not perform as expected.')

    def test_write_ldpc_params(self):
        with TemporaryDirectory() as tmp_dir:
            parity_check_matrix = choice((0, 1), (720, 1440))

            file_path = tmp_dir + '/matrix.txt'
            write_ldpc_params(parity_check_matrix, file_path)
            assert_equal(get_ldpc_code_params(file_path, True)['parity_check_matrix'].toarray(), parity_check_matrix,
                         'The loaded matrix is not equal to the written one.')

    def test_triang_ldpc_systematic_encode(self):
        ldpc_design_files = (os.path.join(self.dir, '../designs/ldpc/wimax/1440.720.txt'),
                             os.path.join(self.dir, '../designs/ldpc/wimax/960.720.a.txt'))
        wimax_ldpc_params = [get_ldpc_code_params(ldpc_design_file) for ldpc_design_file in ldpc_design_files]

        for param in wimax_ldpc_params:
            # Test padding
            with assert_raises(ValueError):
                triang_ldpc_systematic_encode(choice((0, 1), 2), param, False)
            triang_ldpc_systematic_encode(choice((0, 1), 720), param, False)

            # Test encoding
            message_bits = choice((0, 1), 1450)
            coded_bits = triang_ldpc_systematic_encode(message_bits, param)
            syndrome = param['parity_check_matrix'].dot(coded_bits).reshape(-1) % 2
            assert_allclose(syndrome, zeros_like(syndrome), err_msg='Coded message is not in the code book.')

            # Test decoding
            coded_bits[coded_bits == 1] = -1
            coded_bits[coded_bits == 0] = 1
            MSA_decoded_bits = ldpc_bp_decode(coded_bits.reshape(-1, order='F').astype(float), param, 'MSA', 10)[0]
            SPA_decoded_bits = ldpc_bp_decode(coded_bits.reshape(-1, order='F').astype(float), param, 'SPA', 10)[0]

            # Extract systematic part
            MSA_decoded_bits = MSA_decoded_bits[:720].reshape(-1, order='F')
            SPA_decoded_bits = SPA_decoded_bits[:720].reshape(-1, order='F')
            assert_equal(MSA_decoded_bits[:len(message_bits)], message_bits,
                         'Encoded and decoded messages do not match the initial bits without noise (MS algorithm)')
            assert_equal(SPA_decoded_bits[:len(message_bits)], message_bits,
                         'Encoded and decoded messages do not match the initial bits without noise (SP algorithm)')
