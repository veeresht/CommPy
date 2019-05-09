# Authors: Bastien Trotobas <bastien.trotobas@gmail.com>
# License: BSD 3-Clause

from __future__ import division  # Python 2 compatibility

from numpy import arange, sqrt, identity, zeros, log10
from numpy.random import seed
from numpy.testing import run_module_suite, assert_allclose, dec
from scipy.special import erfc

from commpy.channels import MIMOFlatChannel, SISOFlatChannel
from commpy.modulation import QAMModem, kbest
from commpy.utilities import link_performance


@dec.slow
def test_link_performance():
    # Apply link_performance to SISO QPSK and AWGN channel
    BERs = link_performance(QAMModem(4), SISOFlatChannel(fading_param=(1 + 0j, 0)), None,
                            range(0, 9, 2), 600e4, 600)
    desired = erfc(sqrt(10**(arange(0, 9, 2) / 10) / 2)) / 2
    assert_allclose(BERs, desired, rtol=0.25, err_msg='Wrong performance for SISO QPSK and AWGN channel')

    # Apply link_performance to MIMO 16QAM and 4x4 Rayleigh channel
    RayleighChannel = MIMOFlatChannel(4, 4, None, (zeros((4, 4), dtype=complex), identity(4), identity(4)))
    SNRs = range(0, 21, 5) + 10 * log10(4)

    def kbest16(y, h, constellation):
        return kbest(y, h, constellation, 16)

    BERs = link_performance(QAMModem(16), RayleighChannel, kbest16,
                            SNRs, 600e4, 600)
    desired = (2e-1, 1e-1, 3e-2, 2e-3, 4e-5)  # From reference
    assert_allclose(BERs, desired, rtol=1, err_msg='Wrong performance for MIMO 16QAM and 4x4 Rayleigh channel')


if __name__ == "__main__":
    seed(17121996)
    run_module_suite()
