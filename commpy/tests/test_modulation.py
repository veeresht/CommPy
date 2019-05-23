# Authors: Youness Akourim <akourim97@gmail.com>
# License: BSD 3-Clause

from commpy.channels import MIMOFlatChannel
from commpy.links import *
from commpy.modulation import QAMModem, mimo_ml, bit_lvl_repr
from numpy import zeros, identity, arange, concatenate, log2, array
from numpy.random import seed
from numpy.testing import run_module_suite, assert_allclose, dec


@dec.slow
def test_bit_lvl_repr():
    qam = QAMModem(4)

    nb_rx = 2
    nb_tx = 2
    RayleighChannel = MIMOFlatChannel(nb_tx, nb_rx)
    RayleighChannel.fading_param = (zeros((nb_rx, nb_tx), complex), identity(nb_tx), identity(nb_rx))

    SNR = arange(10, 16, 5)

    def receiverWithBLR(y, H, cons):
        beta = int(log2(len(cons)))
        # creation de w
        reel = [pow(2, i) for i in range(beta // 2 - 1, -1, -1)]
        im = [1j * pow(2, i) for i in range(beta // 2 - 1, -1, -1)]
        w = concatenate((reel, im), axis=None)
        A = bit_lvl_repr(H, w)
        mes = array(mimo_ml(y, A, [-1, 1]))
        mes[mes == -1] = 0
        return mes

    def receiverWithoutBLR(y, H, cons):
        return qam.demodulate(mimo_ml(y, H, cons), 'hard')

    MymodelWithoutBLR = \
        linkModel(qam.modulate, RayleighChannel, receiverWithoutBLR, qam.num_bits_symbol, qam.constellation, qam.Es)
    MymodelWithBLR = \
        linkModel(qam.modulate, RayleighChannel, receiverWithBLR, qam.num_bits_symbol, qam.constellation, qam.Es)

    BERWithoutBLR = link_performance(MymodelWithoutBLR, SNR, 300e4, 300)
    BERWithBLR = link_performance(MymodelWithBLR, SNR, 300e4, 300)
    assert_allclose(BERWithoutBLR, BERWithBLR, rtol=0.25,
                    err_msg='bit_lvl_repr changes the performance')


if __name__ == "__main__":
    seed(17121996)
    run_module_suite()
