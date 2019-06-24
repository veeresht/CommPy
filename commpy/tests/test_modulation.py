# Authors: Youness Akourim <akourim97@gmail.com>
# License: BSD 3-Clause

from numpy import zeros, identity, arange, concatenate, log2, array, inf
from numpy.random import seed
from numpy.testing import run_module_suite, assert_allclose, dec

from commpy.channels import MIMOFlatChannel
from commpy.links import *
from commpy.modulation import QAMModem, mimo_ml, bit_lvl_repr, max_log_approx


@dec.slow
def test_bit_lvl_repr():
    # Test the BLR by comparing the performance of a receiver with and without it.

    qam = QAMModem(4)

    nb_rx = 2
    nb_tx = 2
    RayleighChannel = MIMOFlatChannel(nb_tx, nb_rx)
    RayleighChannel.fading_param = (zeros((nb_rx, nb_tx), complex), identity(nb_tx), identity(nb_rx))

    SNR = arange(10, 16, 5)

    def receiver_with_blr(y, H, cons):
        # Create w
        beta = int(log2(len(cons)))
        reel = [pow(2, i) for i in range(beta // 2 - 1, -1, -1)]
        im = [1j * pow(2, i) for i in range(beta // 2 - 1, -1, -1)]
        w = concatenate((reel, im), axis=None)

        # Compute bit level representation
        A = bit_lvl_repr(H, w)
        mes = array(mimo_ml(y, A, [-1, 1]))
        mes[mes == -1] = 0
        return mes

    def receiver_without_blr(y, H, cons):
        return qam.demodulate(mimo_ml(y, H, cons), 'hard')

    my_model_without_blr = \
        LinkModel(qam.modulate, RayleighChannel, receiver_without_blr, qam.num_bits_symbol, qam.constellation, qam.Es)
    my_model_with_blr = \
        LinkModel(qam.modulate, RayleighChannel, receiver_with_blr, qam.num_bits_symbol, qam.constellation, qam.Es)

    ber_without_blr = link_performance(my_model_without_blr, SNR, 300e4, 300)
    ber_with_blr = link_performance(my_model_with_blr, SNR, 300e4, 300)
    assert_allclose(ber_without_blr, ber_with_blr, rtol=0.5,
                    err_msg='bit_lvl_repr changes the performance')


def test_max_log_approx():
    x = array((-1, -1, 1))
    H = array(((-0.33, 0.66, 0.03), (1.25, 0.2, -0.4), (0.05, 1.3, 1.4)))
    y = H.dot(x)
    noise = array((0.45, 1, -1.7))
    pts_list = array(((-1, -1, 1), (-1, 1, 1), (1, 1, 1)))

    def decode(pt):
        return QAMModem(4).demodulate(pt, 'hard')

    # noise_var = 1
    LLR = max_log_approx(y + noise, H, 1, pts_list, decode)
    assert_allclose(LLR, (-9.45, -inf, -7.75, -inf, inf, -inf), atol=0.1,
                    err_msg='Wrong LLRs with noise')

    # noise_var = 0
    LLR = max_log_approx(y, H, 0, pts_list, decode)
    assert_allclose(LLR, (-inf, -inf, -inf, -inf, inf, -inf),
                    err_msg='Wrong LLRs without noise')


if __name__ == "__main__":
    seed(17121996)
    run_module_suite()
