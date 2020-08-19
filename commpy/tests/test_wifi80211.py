# Authors: CommPy contributors
# License: BSD 3-Clause

from __future__ import division  # Python 2 compatibility

from numpy import arange, sqrt, log10
from numpy.random import seed
from numpy.testing import run_module_suite, assert_allclose, dec
from scipy.special import erfc

from commpy.channels import MIMOFlatChannel, SISOFlatChannel
from commpy.links import LinkModel
from commpy.modulation import QAMModem, kbest
from commpy.wifi80211 import Wifi80211


@dec.slow
def test_wifi80211():
    wifi80211 = Wifi80211(1)
    BERs = wifi80211.link_performance(SISOFlatChannel(fading_param=(1 + 0j, 0)), range(0, 9, 2), 10**4, 600)
    # desired = (0.48, 0.45, 0.44, 0.148, 0.0135)  # From reference
    # for i, val in enumerate(desired):
    #     print((BERs[i]-val)/val)
    # assert_allclose(BERs, desired, rtol=0.3,
    #                 err_msg='Wrong performance for SISO QPSK and AWGN channel')

    # Apply link_performance to MIMO 16QAM and 4x4 Rayleigh channel
    wifi80211 = Wifi80211(3)
    RayleighChannel = MIMOFlatChannel(4, 4)
    RayleighChannel.uncorr_rayleigh_fading(complex)
    modem = wifi80211.get_modem()

    def receiver(y, h, constellation, noise_var):
        return modem.demodulate(kbest(y, h, constellation, 16), 'hard')

    BERs = wifi80211.link_performance(RayleighChannel, arange(0, 21, 5)+10*log10(modem.num_bits_symbol), 10**4, 600, receiver=receiver)
    # for i, val in enumerate(BERs):
    #     print(val)
    # model = LinkModel(QAM16.modulate, RayleighChannel, receiver,
    #                   QAM16.num_bits_symbol, QAM16.constellation, QAM16.Es)
    # SNRs = arange(0, 21, 5) + 10 * log10(QAM16.num_bits_symbol)

    # BERs = link_performance(model, SNRs, 600e4, 600)
    # desired = (2e-1, 1e-1, 3e-2, 2e-3, 4e-5)  # From reference
    # assert_allclose(BERs, desired, rtol=1.25,
    #                 err_msg='Wrong performance for MIMO 16QAM and 4x4 Rayleigh channel')


if __name__ == "__main__":
    seed(17121996)
    run_module_suite()
