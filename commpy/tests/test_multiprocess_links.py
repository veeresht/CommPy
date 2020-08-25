# Authors: CommPy contributors
# License: BSD 3-Clause

from __future__ import division  # Python 2 compatibility

from numpy import arange, sqrt, log10
from numpy.random import seed
from numpy.testing import run_module_suite, assert_allclose, dec
from scipy.special import erfc

from commpy.channels import MIMOFlatChannel, SISOFlatChannel
from commpy.modulation import QAMModem, kbest
from commpy.multiprocess_links import LinkModel, Wifi80211

# from commpy.tests.test_multiprocess_links_support import QPSK, receiver

QPSK = QAMModem(4)


def receiver(y, h, constellation, noise_var):
    return QPSK.demodulate(y, 'hard')


QAM16 = QAMModem(16)


def receiver16(y, h, constellation, noise_var):
    return QAM16.demodulate(kbest(y, h, constellation, 16), 'hard')


@dec.slow
def test_link_performance():
    # Set seed
    seed(17121996)

    # Apply link_performance to SISO QPSK and AWGN channel

    model = LinkModel(QPSK.modulate, SISOFlatChannel(fading_param=(1 + 0j, 0)), receiver,
                      QPSK.num_bits_symbol, QPSK.constellation, QPSK.Es)

    BERs = model.link_performance(range(0, 9, 2), 600e4, 600)
    desired = erfc(sqrt(10 ** (arange(0, 9, 2) / 10) / 2)) / 2
    assert_allclose(BERs, desired, rtol=0.25,
                    err_msg='Wrong performance for SISO QPSK and AWGN channel')
    full_metrics = model.link_performance_full_metrics(range(0, 9, 2), 1000, 600)
    assert_allclose(full_metrics[0], desired, rtol=0.25,
                    err_msg='Wrong performance for SISO QPSK and AWGN channel')

    # Apply link_performance to MIMO 16QAM and 4x4 Rayleigh channel
    RayleighChannel = MIMOFlatChannel(4, 4)
    RayleighChannel.uncorr_rayleigh_fading(complex)

    model = LinkModel(QAM16.modulate, RayleighChannel, receiver16,
                      QAM16.num_bits_symbol, QAM16.constellation, QAM16.Es)
    SNRs = arange(0, 21, 5) + 10 * log10(QAM16.num_bits_symbol)

    BERs = model.link_performance(SNRs, 600e4, 600)
    desired = (2e-1, 1e-1, 3e-2, 2e-3, 4e-5)  # From reference
    assert_allclose(BERs, desired, rtol=1.25,
                    err_msg='Wrong performance for MIMO 16QAM and 4x4 Rayleigh channel')
    full_metrics = model.link_performance_full_metrics(SNRs, 1000, 600)
    assert_allclose(full_metrics[0], desired, rtol=1.25,
                    err_msg='Wrong performance for MIMO 16QAM and 4x4 Rayleigh channel')


@dec.slow
def test_wifi80211_siso_channel():
    seed(17121996)
    wifi80211 = Wifi80211(1)
    BERs = wifi80211.link_performance(SISOFlatChannel(fading_param=(1 + 0j, 0)), range(0, 9, 2), 10 ** 4, 600)[0]
    desired = (0.489, 0.503, 0.446, 0.31, 0.015)  # From previous tests
    # for i, val in enumerate(desired):
    #     print((BERs[i] - val) / val)
    assert_allclose(BERs, desired, rtol=0.3,
                    err_msg='Wrong performance for SISO QPSK and AWGN channel')


wifi80211 = Wifi80211(3)
modem = wifi80211.get_modem()


def receiver_mimo_wifi3(y, h, constellation, noise_var):
    return modem.demodulate(kbest(y, h, constellation, 16), 'hard')


@dec.slow
def test_wifi80211_mimo_channel():
    seed(17121996)
    # Apply link_performance to MIMO 16QAM and 4x4 Rayleigh channel
    RayleighChannel = MIMOFlatChannel(4, 4)
    RayleighChannel.uncorr_rayleigh_fading(complex)

    BERs = wifi80211.link_performance(RayleighChannel, arange(0, 21, 5) + 10 * log10(modem.num_bits_symbol), 10 ** 4,
                                      600, receiver=receiver_mimo_wifi3)[0]
    desired = (0.535, 0.508, 0.521, 0.554, 0.475)  # From previous test
    assert_allclose(BERs, desired, rtol=1.25,
                    err_msg='Wrong performance for MIMO 16QAM and 4x4 Rayleigh channel')


if __name__ == "__main__":
    run_module_suite()
