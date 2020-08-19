# Authors: CommPy contributors
# License: BSD 3-Clause

from __future__ import division  # Python 2 compatibility

from numpy import arange, log10
from numpy.random import seed
from numpy.testing import run_module_suite, dec, assert_allclose

from commpy.channels import MIMOFlatChannel, SISOFlatChannel
from commpy.modulation import kbest
from commpy.wifi80211 import Wifi80211


@dec.slow
def test_wifi80211_siso_channel():
    seed(17121996)
    wifi80211 = Wifi80211(1)
    BERs = wifi80211.link_performance(SISOFlatChannel(fading_param=(1 + 0j, 0)), range(0, 9, 2), 10 ** 4, 600)
    desired = (0.489, 0.503, 0.446, 0.31, 0.015)  # From previous tests
    for i, val in enumerate(desired):
        print((BERs[i] - val) / val)
    assert_allclose(BERs, desired, rtol=0.3,
                    err_msg='Wrong performance for SISO QPSK and AWGN channel')


@dec.slow
def test_wifi80211_mimo_channel():
    seed(17121996)
    # Apply link_performance to MIMO 16QAM and 4x4 Rayleigh channel
    wifi80211 = Wifi80211(3)
    RayleighChannel = MIMOFlatChannel(4, 4)
    RayleighChannel.uncorr_rayleigh_fading(complex)
    modem = wifi80211.get_modem()

    def receiver(y, h, constellation, noise_var):
        return modem.demodulate(kbest(y, h, constellation, 16), 'hard')

    BERs = wifi80211.link_performance(RayleighChannel, arange(0, 21, 5) + 10 * log10(modem.num_bits_symbol), 10 ** 4,
                                      600, receiver=receiver)
    desired = (0.535, 0.508, 0.521, 0.554, 0.475)  # From previous test
    assert_allclose(BERs, desired, rtol=1.25,
                    err_msg='Wrong performance for MIMO 16QAM and 4x4 Rayleigh channel')


if __name__ == "__main__":
    run_module_suite()
