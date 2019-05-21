# Authors: Bastien Trotobas <bastien.trotobas@gmail.com>
# License: BSD 3-Clause

from __future__ import division  # Python 2 compatibility

from numpy import arange, sqrt, log10
from numpy.random import seed
from numpy.testing import run_module_suite, assert_allclose, dec
from scipy.special import erfc

from commpy.channels import MIMOFlatChannel, SISOFlatChannel
from commpy.links import link_performance, linkModel
from commpy.modulation import QAMModem, kbest


@dec.slow
def test_link_performance():
    # Apply link_performance to SISO QPSK and AWGN channel
    QPSK = QAMModem(4)

    def receiver(y, h, constellation):
        return QPSK.demodulate(y, 'hard')
    model = linkModel(QPSK.modulate, SISOFlatChannel(fading_param=(1 + 0j, 0)), receiver,
                      QPSK.num_bits_symbol, QPSK.constellation, QPSK.Es)

    BERs = link_performance(model, range(0, 9, 2), 600e4, 600)
    desired = erfc(sqrt(10**(arange(0, 9, 2) / 10) / 2)) / 2
    assert_allclose(BERs, desired, rtol=0.25,
                    err_msg='Wrong performance for SISO QPSK and AWGN channel')

    # Apply link_performance to MIMO 16QAM and 4x4 Rayleigh channel
    QAM16 = QAMModem(16)
    RayleighChannel = MIMOFlatChannel(4, 4)
    RayleighChannel.uncorr_rayleigh_fading(complex)

    def receiver(y, h, constellation):
        return QAM16.demodulate(kbest(y, h, constellation, 16), 'hard')
    model = linkModel(QAM16.modulate, RayleighChannel, receiver,
                      QAM16.num_bits_symbol, QAM16.constellation, QAM16.Es)
    SNRs = arange(0, 21, 5) + 10 * log10(QAM16.num_bits_symbol)

    BERs = link_performance(model, SNRs, 600e4, 600)
    desired = (2e-1, 1e-1, 3e-2, 2e-3, 4e-5)  # From reference
    assert_allclose(BERs, desired, rtol=1.25,
                    err_msg='Wrong performance for MIMO 16QAM and 4x4 Rayleigh channel')


if __name__ == "__main__":
    seed(17121996)
    run_module_suite()
