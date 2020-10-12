# Authors: CommPy contributors
# License: BSD 3-Clause

from __future__ import division  # Python 2 compatibility

from numpy import arange, sqrt, log10
from numpy.random import seed
from numpy.testing import run_module_suite, assert_allclose, dec
from scipy.special import erfc

from commpy.channelcoding.ldpc import get_ldpc_code_params, triang_ldpc_systematic_encode, ldpc_bp_decode
from commpy.channels import MIMOFlatChannel, SISOFlatChannel
from commpy.links import link_performance, LinkModel
from commpy.modulation import QAMModem, kbest, best_first_detector


@dec.slow
def test_link_performance():
    # Set seed
    seed(8071996)
    ######################################
    # Build models & desired solutions
    ######################################
    models = []
    desired_bers = []
    snr_range = []
    labels = []
    rtols = []
    code_rates = []

    # SISO QPSK and AWGN channel
    QPSK = QAMModem(4)

    def receiver(y, h, constellation, noise_var):
        return QPSK.demodulate(y, 'hard')

    models.append(LinkModel(QPSK.modulate, SISOFlatChannel(fading_param=(1 + 0j, 0)), receiver,
                            QPSK.num_bits_symbol, QPSK.constellation, QPSK.Es))
    snr_range.append(arange(0, 9, 2))
    desired_bers.append(erfc(sqrt(10 ** (snr_range[-1] / 10) / 2)) / 2)
    labels.append('SISO QPSK and AWGN channel')
    rtols.append(.25)
    code_rates.append(1)

    # MIMO 16QAM, 4x4 Rayleigh channel and hard-output K-Best
    QAM16 = QAMModem(16)
    RayleighChannel = MIMOFlatChannel(4, 4)
    RayleighChannel.uncorr_rayleigh_fading(complex)

    def receiver(y, h, constellation, noise_var):
        return QAM16.demodulate(kbest(y, h, constellation, 16), 'hard')

    models.append(LinkModel(QAM16.modulate, RayleighChannel, receiver,
                            QAM16.num_bits_symbol, QAM16.constellation, QAM16.Es))
    snr_range.append(arange(0, 21, 5) + 10 * log10(QAM16.num_bits_symbol))
    desired_bers.append((2e-1, 1e-1, 3e-2, 2e-3, 4e-5))  # From reference
    labels.append('MIMO 16QAM, 4x4 Rayleigh channel and hard-output K-Best')
    rtols.append(1.25)
    code_rates.append(1)

    # MIMO 16QAM, 4x4 Rayleigh channel and soft-output best-first
    QAM16 = QAMModem(16)
    RayleighChannel = MIMOFlatChannel(4, 4)
    RayleighChannel.uncorr_rayleigh_fading(complex)
    ldpc_params = get_ldpc_code_params('commpy/channelcoding/designs/ldpc/wimax/1440.720.txt', True)

    def modulate(bits):
        return QAM16.modulate(triang_ldpc_systematic_encode(bits, ldpc_params, False).reshape(-1, order='F'))

    def decoder(llrs):
        return ldpc_bp_decode(llrs, ldpc_params, 'MSA', 15)[0][:720].reshape(-1, order='F')

    def demode(symbs):
        return QAM16.demodulate(symbs, 'hard')

    def receiver(y, h, constellation, noise_var):
        return best_first_detector(y, h, constellation, (1, 3, 5), noise_var, demode, 500)

    models.append(LinkModel(modulate, RayleighChannel, receiver,
                            QAM16.num_bits_symbol, QAM16.constellation, QAM16.Es,
                            decoder, 0.5))
    snr_range.append(arange(17, 20, 1))
    desired_bers.append((1.7e-1, 1e-1, 2.5e-3))  # From reference
    labels.append('MIMO 16QAM, 4x4 Rayleigh channel and soft-output best-first')
    rtols.append(2)
    code_rates.append(.5)

    ######################################
    # Make tests
    ######################################

    for test in range(len(models)):
        BERs = link_performance(models[test], snr_range[test], 5e5, 200, 720, models[test].rate)
        assert_allclose(BERs, desired_bers[test], rtol=rtols[test],
                        err_msg='Wrong performance for ' + labels[test])
        full_metrics = models[test].link_performance_full_metrics(snr_range[test], 2500, 200, 720, models[test].rate)
        assert_allclose(full_metrics[0], desired_bers[test], rtol=rtols[test],
                        err_msg='Wrong performance for ' + labels[test])


if __name__ == "__main__":
    run_module_suite()
