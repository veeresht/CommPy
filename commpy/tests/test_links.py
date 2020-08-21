# Authors: CommPy contributors
# License: BSD 3-Clause

from __future__ import division  # Python 2 compatibility

from numpy import arange, sqrt, log10
from numpy.random import seed
from numpy.testing import run_module_suite, assert_allclose, dec
from scipy.special import erfc

from commpy.channels import MIMOFlatChannel, SISOFlatChannel
from commpy.links import link_performance, LinkModel
from commpy.modulation import QAMModem, kbest


@dec.slow
def test_link_performance():
    # Set seed
    seed(17121996)

    # Apply link_performance to SISO QPSK and AWGN channel
    QPSK = QAMModem(4)

    def receiver(y, h, constellation, noise_var):
        return QPSK.demodulate(y, 'hard')

    model = LinkModel(QPSK.modulate, SISOFlatChannel(fading_param=(1 + 0j, 0)), receiver,
                      QPSK.num_bits_symbol, QPSK.constellation, QPSK.Es)

    BERs = link_performance(model, range(0, 9, 2), 600e4, 600)
    desired = erfc(sqrt(10 ** (arange(0, 9, 2) / 10) / 2)) / 2
    assert_allclose(BERs, desired, rtol=0.25,
                    err_msg='Wrong performance for SISO QPSK and AWGN channel')
    full_metrics = model.link_performance_full_metrics(range(0, 9, 2), 1000, 600)
    assert_allclose(full_metrics[0], desired, rtol=0.25,
                    err_msg='Wrong performance for SISO QPSK and AWGN channel')

    # Apply link_performance to MIMO 16QAM and 4x4 Rayleigh channel
    QAM16 = QAMModem(16)
    RayleighChannel = MIMOFlatChannel(4, 4)
    RayleighChannel.uncorr_rayleigh_fading(complex)

    def receiver(y, h, constellation, noise_var):
        return QAM16.demodulate(kbest(y, h, constellation, 16), 'hard')

    model = LinkModel(QAM16.modulate, RayleighChannel, receiver,
                      QAM16.num_bits_symbol, QAM16.constellation, QAM16.Es)
    SNRs = arange(0, 21, 5) + 10 * log10(QAM16.num_bits_symbol)

    BERs = link_performance(model, SNRs, 600e4, 600)
    desired = (2e-1, 1e-1, 3e-2, 2e-3, 4e-5)  # From reference
    assert_allclose(BERs, desired, rtol=1.25,
                    err_msg='Wrong performance for MIMO 16QAM and 4x4 Rayleigh channel')
    full_metrics = model.link_performance_full_metrics(SNRs, 1000, 600)
    assert_allclose(full_metrics[0], desired, rtol=1.25,
                    err_msg='Wrong performance for MIMO 16QAM and 4x4 Rayleigh channel')


if __name__ == "__main__":
    run_module_suite()
