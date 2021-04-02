# Authors: CommPy contributors
# License: BSD 3-Clause

from itertools import product

from numpy import zeros, identity, arange, concatenate, log2, log10, array, inf, sqrt, sin, pi
from numpy.random import seed
from numpy.testing import run_module_suite, assert_allclose, dec, assert_raises, assert_array_equal
from scipy.special import erf

from commpy.channels import MIMOFlatChannel, SISOFlatChannel
from commpy.links import *
from commpy.modulation import QAMModem, mimo_ml, bit_lvl_repr, max_log_approx, PSKModem, Modem
from commpy.utilities import signal_power


def Qfunc(x):
    return 0.5 - 0.5 * erf(x / sqrt(2))


@dec.slow
def test_bit_lvl_repr():
    # Set seed
    seed(17121996)

    # Test the BLR by comparing the performance of a receiver with and without it.
    qam = QAMModem(4)

    nb_rx = 2
    nb_tx = 2
    RayleighChannel = MIMOFlatChannel(nb_tx, nb_rx)
    RayleighChannel.fading_param = (zeros((nb_rx, nb_tx), complex), identity(nb_tx), identity(nb_rx))

    SNR = arange(10, 16, 5)

    def receiver_with_blr(y, H, cons, noise_var):
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

    def receiver_without_blr(y, H, cons, noise_var):
        return qam.demodulate(mimo_ml(y, H, cons), 'hard')

    my_model_without_blr = \
        LinkModel(qam.modulate, RayleighChannel, receiver_without_blr, qam.num_bits_symbol, qam.constellation, qam.Es)
    my_model_with_blr = \
        LinkModel(qam.modulate, RayleighChannel, receiver_with_blr, qam.num_bits_symbol, qam.constellation, qam.Es)

    ber_without_blr = link_performance(my_model_without_blr, SNR, 300e4, 300)
    ber_with_blr = link_performance(my_model_with_blr, SNR, 300e4, 300)
    assert_allclose(ber_without_blr, ber_with_blr, rtol=0.5,
                    err_msg='bit_lvl_repr changes the performance')

    # Test error raising
    with assert_raises(ValueError):
        bit_lvl_repr(RayleighChannel.channel_gains[0], array((2, 4, 6)))


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
    assert_allclose(LLR, (9.45, inf, 7.75, inf, -inf, inf), atol=0.1,
                    err_msg='Wrong LLRs with noise')

    # noise_var = 0
    LLR = max_log_approx(y, H, 0, pts_list, decode)
    assert_allclose(LLR, (inf, inf, inf, inf, -inf, inf),
                    err_msg='Wrong LLRs without noise')


class ModemTestcase:
    qam_modems = [QAMModem(4), QAMModem(16), QAMModem(64)]
    psk_modems = [PSKModem(4), PSKModem(16), PSKModem(64)]
    modems = qam_modems + psk_modems

    def __init__(self):
        # Create a custom Modem
        custom_constellation = [re + im * 1j for re, im in product((-3.5, -0.5, 0.5, 3.5), repeat=2)]
        self.custom_modems = [Modem(custom_constellation)]

        # Add to custom modems a QAM modem with modified constellation
        QAM_custom = QAMModem(16)
        QAM_custom.constellation = custom_constellation
        self.custom_modems.append(QAM_custom)
        self.modems += self.custom_modems

        # Assert that error is raised when the contellation length is not a power of 2
        with assert_raises(ValueError):
            QAM_custom.constellation = (0, 0, 0)

    def test(self):
        for modem in self.modems:
            self.do(modem)
        for modem in self.qam_modems:
            self.do_qam(modem)
        for modem in self.psk_modems:
            self.do_psk(modem)
        for modem in self.custom_modems:
            self.do_custom(modem)

    # Default methods for TestClasses that not implement a specific test
    def do(self, modem):
        pass

    def do_qam(self, modem):
        pass

    def do_psk(self, modem):
        pass

    def do_custom(self, modem):
        pass


@dec.slow
class TestModulateHardDemodulate(ModemTestcase):

    @staticmethod
    def check_BER(modem, EbN0dB, BERs_expected):
        seed(8071996)
        model = LinkModel(modem.modulate,
                          SISOFlatChannel(fading_param=(1 + 0j, 0)),
                          lambda y, _, __, ___: modem.demodulate(y, 'hard'),
                          modem.num_bits_symbol, modem.constellation, modem.Es)
        BERs = model.link_performance(EbN0dB + 10 * log10(log2(modem.m)), 5e5, 400, 720)
        assert_allclose(BERs, BERs_expected, atol=1e-4, rtol=.1,
                        err_msg='Wrong BER for a standard modulation with {} symbols'.format(modem.m))

    def do_qam(self, modem):
        EbN0dB = arange(8, 25, 4)
        nb_symb_pam = sqrt(modem.m)
        BERs_expected = 2 * (1 - 1 / nb_symb_pam) / log2(nb_symb_pam) * \
                        Qfunc(sqrt(3 * log2(nb_symb_pam) / (nb_symb_pam ** 2 - 1) * (2 * 10 ** (EbN0dB / 10))))
        self.check_BER(modem, EbN0dB, BERs_expected)

    def do_psk(self, modem):
        EbN0dB = arange(15, 25, 4)
        SERs_expected = 2 * Qfunc(sqrt(2 * modem.num_bits_symbol * 10 ** (EbN0dB / 10)) * sin(pi / modem.m))
        BERs_expected = SERs_expected / modem.num_bits_symbol
        self.check_BER(modem, EbN0dB, BERs_expected)

    def do(self, modem):
        for bits in product(*((0, 1),) * modem.num_bits_symbol):
            assert_array_equal(bits, modem.demodulate(modem.modulate(bits), 'hard'),
                               err_msg='Bits are not equal after modulation and hard demodulation')


class TestEs(ModemTestcase):

    def do_qam(self, modem):
        assert_allclose(signal_power(modem.constellation), 2 * (modem.m - 1) / 3)

    def do_psk(self, modem):
        assert_allclose(modem.Es, 1)

    def do_custom(self, modem):
        assert_allclose(modem.Es, 12.5)


if __name__ == "__main__":
    run_module_suite()
