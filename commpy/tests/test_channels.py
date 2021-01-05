# Authors: CommPy contributors
# License: BSD 3-Clause

from __future__ import division, print_function  # Python 2 compatibility

from math import cos

from numpy import ones, inf, sqrt, array, identity, zeros, dot, trace, einsum, absolute, exp, pi, fromiter, kron, \
    zeros_like, empty
from numpy.random import seed, choice, randn
from numpy.testing import run_module_suite, assert_raises, assert_equal, assert_allclose, \
    assert_array_equal, dec

from commpy.channels import SISOFlatChannel, MIMOFlatChannel
from commpy.utilities import signal_power


class TestSISOFlatChannel:
    msg_length = 100000
    real_mods = array((-1, 1)), array((-3, 3))
    all_mods = array((-1, 1)), array((-3, 3)), \
               array((-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j)), array((-3 - 3j, -3 + 3j, 3 - 3j, 3 + 3j))

    def test_default_args(self):
        def check(chan):
            assert_equal(chan.noises, None,
                         err_msg='Default noises is not None')
            assert_equal(chan.channel_gains, None,
                         err_msg='Default channel gains is not None')
            assert_equal(chan.unnoisy_output, None,
                         err_msg='Default unnoisy output is not None')

        chan = SISOFlatChannel()

        # Test output state before any propagation
        check(chan)

        # Test that noise standard deviation must be set before propagation
        with assert_raises(AssertionError):
            chan.propagate(array((1, 1)))

        # Test output state before any propagation
        check(chan)

        assert_equal(chan.nb_rx, 1,
                     err_msg='SISO channel as more than 1 Rx')
        assert_equal(chan.nb_tx, 1,
                     err_msg='SISO channel as more than 1 Tx')

    def test_fading(self):
        # Set seed
        seed(17121996)

        def check_chan_gain(mod, chan):
            msg = choice(mod, self.msg_length)
            chan.propagate(msg)

            P_msg = signal_power(msg)
            P_unnoisy = signal_power(chan.unnoisy_output)

            assert_allclose(P_unnoisy, P_msg, rtol=0.2,
                            err_msg='Channel add or remove energy')

        # Test value checking in constructor construction
        with assert_raises(ValueError):
            SISOFlatChannel(0, (1, 1))

        chan = SISOFlatChannel(0)

        # Test on real channel
        for mod in self.real_mods:
            # Test value checking after construction
            with assert_raises(ValueError):
                chan.fading_param = (1, 1)

            # Test without fading
            chan.fading_param = (1, 0)
            check_chan_gain(mod, chan)
            assert_array_equal(chan.channel_gains, ones(self.msg_length),
                               err_msg='Channel fading while fading is disabled')

            # Test with Rayleigh fading
            chan.fading_param = (0, 1)
            check_chan_gain(mod, chan)
            assert_allclose(absolute(chan.channel_gains.mean()), 0, atol=2e-2,
                            err_msg='Wrong channel mean with real channel')
            assert_allclose(chan.channel_gains.var(), 1, atol=0.2,
                            err_msg='Wrong channel variance with real channel')

            # Test with rician fading
            chan.fading_param = (sqrt(2 / 3), 1 / 3)
            check_chan_gain(mod, chan)
            assert_allclose(chan.channel_gains.mean(), sqrt(2 / 3), atol=2e-2,
                            err_msg='Wrong channel mean with real channel')
            assert_allclose(chan.channel_gains.var(), 1 / 3, atol=0.2,
                            err_msg='Wrong channel variance with real channel')

        # Test on complex channel
        for mod in self.all_mods:
            # Test value checking after construction
            with assert_raises(ValueError):
                chan.fading_param = (1, 1)

            # Test without fading
            chan.fading_param = (1 + 0j, 0)
            check_chan_gain(mod, chan)
            assert_array_equal(chan.channel_gains, ones(self.msg_length),
                               err_msg='Channel fading while fading is disabled')

            # Test with Rayleigh fading
            chan.fading_param = (0j, 1)
            check_chan_gain(mod, chan)
            assert_allclose(absolute(chan.channel_gains.mean()), 0, atol=2e-2,
                            err_msg='Wrong channel mean with real channel')
            assert_allclose(chan.channel_gains.var(), 1, atol=0.2,
                            err_msg='Wrong channel variance with real channel')

            # Test with rician fading
            chan.fading_param = (0.5 + 0.5j, 0.5)
            check_chan_gain(mod, chan)
            assert_allclose(absolute(chan.channel_gains.mean()), sqrt(0.5), atol=2e-2,
                            err_msg='Wrong channel mean with real channel')
            assert_allclose(chan.channel_gains.var(), 0.5, atol=0.2,
                            err_msg='Wrong channel variance with real channel')

    def test_noise_generation(self):
        # Set seed
        seed(17121996)

        def check_noise(mod, chan, corrected_SNR_lin):
            msg = choice(mod, self.msg_length)
            chan.propagate(msg)

            P_msg = signal_power(msg)  # previous test asserted that channel neither add nor remove energy
            P_noise = signal_power(chan.noises)

            assert_allclose(absolute(chan.noises.mean()), 0., atol=5e-2,
                            err_msg='Noise mean is not 0')
            if corrected_SNR_lin == inf:
                assert_allclose(P_noise, 0, atol=1e-2,
                                err_msg='There is noise that should not be here')
            else:
                assert_allclose(P_msg / P_noise, corrected_SNR_lin, atol=0.2,
                                err_msg='Wrong SNR')

        chan = SISOFlatChannel(fading_param=(1 + 0j, 0))
        for mod in self.all_mods:
            chan.noise_std = 0
            check_noise(mod, chan, inf)
            chan.set_SNR_lin(6, Es=signal_power(mod))
            check_noise(mod, chan, 6)
            chan.set_SNR_lin(6, .5, signal_power(mod))
            check_noise(mod, chan, 3)
            chan.set_SNR_dB(0, Es=signal_power(mod))
            check_noise(mod, chan, 1)
            chan.set_SNR_dB(0, .5, signal_power(mod))
            check_noise(mod, chan, .5)

        chan = SISOFlatChannel(fading_param=(1, 0))
        for mod in self.real_mods:
            chan.noise_std = 0
            check_noise(mod, chan, inf)
            chan.set_SNR_lin(6, Es=signal_power(mod))
            check_noise(mod, chan, 6)
            chan.set_SNR_lin(6, .5, signal_power(mod))
            check_noise(mod, chan, 3)
            chan.set_SNR_dB(0, Es=signal_power(mod))
            check_noise(mod, chan, 1)
            chan.set_SNR_dB(0, .5, signal_power(mod))
            check_noise(mod, chan, .5)

    def test_type_check(self):
        chan = SISOFlatChannel(0)
        with assert_raises(TypeError):
            chan.propagate(array((1, 1j)))

    def test_k_factor(self):
        # Real channel
        chan = SISOFlatChannel()
        assert_allclose(chan.k_factor, inf,
                        err_msg='k-factor should be infinite without fading in SISO channels')
        chan.fading_param = 0, 1
        assert_allclose(chan.k_factor, 0,
                        err_msg='k-factor should be 0 with Rayleigh fading in SISO channels')
        chan.fading_param = sqrt(0.5), 0.5
        assert_allclose(chan.k_factor, 1,
                        err_msg='Wrong k-factor with rician fading in SISO channels')

        # Complex channel
        chan.fading_param = 1j, 0
        assert_allclose(chan.k_factor, inf,
                        err_msg='k-factor should be infinite without fading in SISO channels')
        chan.fading_param = 0j, 1
        assert_allclose(chan.k_factor, 0,
                        err_msg='k-factor should be 0 with Rayleigh fading in SISO channels')
        chan.fading_param = 0.5 + 0.5j, 0.5
        assert_allclose(chan.k_factor, 1,
                        err_msg='Wrong k-factor with rician fading in SISO channels')


class MIMOTestCase(object):
    msg_length = 100000
    real_mods = array((-1, 1)), array((-3, 3))
    all_mods = array((-1, 1)), array((-3, 3)), \
               array((-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j)), array((-3 - 3j, -3 + 3j, 3 - 3j, 3 + 3j))

    @staticmethod
    def random_SDP_matrix(n):
        G = randn(n, n)
        dot(G, G.T, G)
        return G / trace(G)

    def test_symetric(self):
        nb_tx = 8
        nb_rx = 8
        self.do(nb_tx, nb_rx)

    def test_more_rx(self):
        nb_tx = 4
        nb_rx = 8
        self.do(nb_tx, nb_rx)

    def test_more_tx(self):
        nb_tx = 8
        nb_rx = 4
        self.do(nb_tx, nb_rx)

    def test_SIMO(self):
        nb_tx = 1
        nb_rx = 8
        self.do(nb_tx, nb_rx)

    def test_MISO(self):
        nb_tx = 8
        nb_rx = 1
        self.do(nb_tx, nb_rx)

    def test_SISO(self):
        nb_tx = 1
        nb_rx = 1
        self.do(nb_tx, nb_rx)


class TestMIMODefaultArgs(MIMOTestCase):
    def __init__(self):
        super(TestMIMODefaultArgs, self).__init__()

    def do(self, nb_tx, nb_rx):
        def check(chan):
            assert_equal(chan.noises, None,
                         err_msg='Default noises is not None')
            assert_equal(chan.channel_gains, None,
                         err_msg='Default channel gains is not None')
            assert_equal(chan.unnoisy_output, None,
                         err_msg='Default unnoisy output is not None')

        chan = MIMOFlatChannel(nb_tx, nb_rx)

        # Test output state before any propagation
        check(chan)

        # Test that noise standard deviation must be set before propagation
        with assert_raises(AssertionError):
            chan.propagate(array((1, 1)))

        # Test output state before any propagation
        check(chan)


@dec.slow
class TestMIMOFading(MIMOTestCase):
    def __init__(self):
        super(TestMIMOFading, self).__init__()

    def do(self, nb_tx, nb_rx):
        # Set seed
        seed(17121996)

        def check_chan_gain(mod, chan):
            msg = choice(mod, self.msg_length)
            chan.propagate(msg)

            P_msg = signal_power(msg)
            P_unnoisy = signal_power(chan.unnoisy_output)

            assert_allclose(P_unnoisy, P_msg * chan.nb_tx, rtol=0.2,
                            err_msg='Channel add or remove energy')

        def expo_correlation(t, r):
            # Construct the exponent matrix
            expo_tx = fromiter((j - i for i in range(chan.nb_tx) for j in range(chan.nb_tx)), int, chan.nb_tx ** 2)
            expo_rx = fromiter((j - i for i in range(chan.nb_rx) for j in range(chan.nb_rx)), int, chan.nb_rx ** 2)

            # Reshape
            expo_tx = expo_tx.reshape(chan.nb_tx, chan.nb_tx)
            expo_rx = expo_rx.reshape(chan.nb_rx, chan.nb_rx)

            return t ** expo_tx, r ** expo_rx

        def check_correlation(chan, Rt, Rr):
            nb_ant = chan.nb_tx * chan.nb_rx
            Rdes = kron(Rt, Rr)
            H = chan.channel_gains
            Ract = zeros_like(Rdes)
            for i in range(len(H)):
                Ract += H[i].T.reshape(nb_ant, 1).dot(H[i].T.reshape(1, nb_ant).conj())
            Ract /= len(H)
            assert_allclose(Rdes, Ract, atol=0.05,
                            err_msg='Wrong correlation matrix')

        # Test value checking in constructor construction
        with assert_raises(ValueError):
            MIMOFlatChannel(nb_tx, nb_tx, 0, (ones((nb_tx, nb_tx)), ones((nb_tx, nb_tx)), ones((nb_rx, nb_rx))))

        chan = MIMOFlatChannel(nb_tx, nb_rx, 0)
        prod_nb = nb_tx * nb_rx

        # Test on real channel
        for mod in self.real_mods:
            # Test value checking after construction
            with assert_raises(ValueError):
                chan.fading_param = (ones((nb_tx, nb_tx)), ones((nb_tx, nb_tx)), ones((nb_rx, nb_rx)))

            # Test with Rayleigh fading
            chan.fading_param = (zeros((nb_rx, nb_tx)), identity(nb_tx), identity(nb_rx))
            check_chan_gain(mod, chan)

            # Test with rician fading
            mean = randn(nb_rx, nb_tx)
            mean *= sqrt(prod_nb * 0.75 / einsum('ij,ij->', absolute(mean), absolute(mean)))
            Rt = self.random_SDP_matrix(nb_tx) * sqrt(prod_nb) * 0.5
            Rr = self.random_SDP_matrix(nb_rx) * sqrt(prod_nb) * 0.5
            chan.fading_param = (mean, Rt, Rr)
            check_chan_gain(mod, chan)

            # Test helper functions
            chan.uncorr_rayleigh_fading(float)
            check_chan_gain(mod, chan)
            assert_allclose(chan.k_factor, 0,
                            err_msg='Wrong k-factor with uncorrelated Rayleigh fading')

            mean = randn(nb_rx, nb_tx)
            chan.uncorr_rician_fading(mean, 10)
            check_chan_gain(mod, chan)
            assert_allclose(chan.k_factor, 10,
                            err_msg='Wrong k-factor with uncorrelated rician fading')

        # Test on complex channel
        for mod in self.all_mods:
            # Test value checking after construction
            with assert_raises(ValueError):
                chan.fading_param = (ones((nb_tx, nb_tx)), ones((nb_tx, nb_tx)), ones((nb_rx, nb_rx)))

            # Test with Rayleigh fading
            chan.fading_param = (zeros((nb_rx, nb_tx), complex), identity(nb_tx), identity(nb_rx))
            check_chan_gain(mod, chan)
            assert_allclose(chan.channel_gains.mean(), 0, atol=1e-2,
                            err_msg='Wrong channel mean with complex channel')
            assert_allclose(chan.channel_gains.var(), 1, atol=5e-2,
                            err_msg='Wrong channel variance with complex channel')

            # Test with rician fading
            mean = randn(nb_rx, nb_tx) + 1j * randn(nb_rx, nb_tx)
            mean *= sqrt(prod_nb * 0.75 / einsum('ij,ij->', absolute(mean), absolute(mean)))
            Rt = self.random_SDP_matrix(nb_tx) * sqrt(prod_nb) * 0.5
            Rr = self.random_SDP_matrix(nb_rx) * sqrt(prod_nb) * 0.5
            chan.fading_param = (mean, Rt, Rr)
            check_chan_gain(mod, chan)

            assert_allclose(chan.channel_gains.mean(0).real, mean.real, atol=0.1,
                            err_msg='Wrong channel mean with complex channel')
            assert_allclose(chan.channel_gains.mean(0).imag, mean.imag, atol=0.1,
                            err_msg='Wrong channel mean with complex channel')

            # Test helper functions
            chan.uncorr_rayleigh_fading(complex)
            check_chan_gain(mod, chan)
            assert_allclose(chan.k_factor, 0,
                            err_msg='Wrong k-factor with uncorrelated Rayleigh fading')

            mean = randn(nb_rx, nb_tx) + randn(nb_rx, nb_tx) * 1j
            chan.uncorr_rician_fading(mean, 10)
            check_chan_gain(mod, chan)
            assert_allclose(chan.k_factor, 10,
                            err_msg='Wrong k-factor with uncorrelated rician fading')

            chan.expo_corr_rayleigh_fading(exp(-0.2j * pi), exp(-0.1j * pi))
            check_chan_gain(mod, chan)
            assert_allclose(chan.k_factor, 0,
                            err_msg='Wrong k-factor with correlated Rayleigh fading')
            Rt, Rr = expo_correlation(exp(-0.2j * pi), exp(-0.1j * pi))
            check_correlation(chan, Rt, Rr)

            mean = randn(nb_rx, nb_tx) + randn(nb_rx, nb_tx) * 1j
            chan.expo_corr_rician_fading(mean, 10, exp(-0.1j * pi), exp(-0.2j * pi))
            check_chan_gain(mod, chan)
            assert_allclose(chan.k_factor, 10,
                            err_msg='Wrong k-factor with correlated rician fading')

            # Test with beta > 0
            chan.expo_corr_rayleigh_fading(exp(-0.2j * pi), exp(-0.1j * pi), 1, 0.5)
            check_chan_gain(mod, chan)
            assert_allclose(chan.k_factor, 0,
                            err_msg='Wrong k-factor with correlated Rayleigh fading')

            mean = randn(nb_rx, nb_tx) + randn(nb_rx, nb_tx) * 1j
            chan.expo_corr_rician_fading(mean, 5, exp(-0.1j * pi), exp(-0.2j * pi), 3, 2)
            check_chan_gain(mod, chan)
            assert_allclose(chan.k_factor, 5,
                            err_msg='Wrong k-factor with correlated rician fading')


class TestMIMOSpectular(MIMOTestCase):
    def __init__(self):
        super(TestMIMOSpectular, self).__init__()

    def do(self, nb_tx, nb_rx):
        chan = MIMOFlatChannel(nb_tx, nb_rx, 0)

        # Test raising of ValueError
        with assert_raises(ValueError):
            chan.specular_compo(0, -1, 0, 1)
        with assert_raises(ValueError):
            chan.specular_compo(0, 1, 0, -1)

        # Test the result
        desired = empty((nb_rx, nb_tx), dtype=complex)
        for n in range(nb_rx):
            for m in range(nb_tx):
                desired[n, m] = exp(1j * 2 * pi * (n * 1 * cos(0.5) - m * 0.1 * cos(2)))
        assert_allclose(chan.specular_compo(2, 0.1, 0.5, 1), desired, rtol=0.02,
                        err_msg='Wrong specular component')


@dec.slow
class TestMIMONoiseGeneration(MIMOTestCase):
    def __init__(self):
        super(TestMIMONoiseGeneration, self).__init__()

    def do(self, nb_tx, nb_rx):
        # Set seed
        seed(17121996)

        def check_noise(mod, chan, corrected_SNR_lin):
            msg = choice(mod, self.msg_length)
            chan.propagate(msg)

            P_msg = signal_power(msg)  # previous test asserted that channel neither add nor remove energy
            P_noise = signal_power(chan.noises)

            assert_allclose(abs(chan.noises.mean()), 0., atol=0.5,
                            err_msg='Noise mean is not 0')
            if corrected_SNR_lin == inf:
                assert_allclose(P_noise, 0, atol=1e-2,
                                err_msg='There is noise that should not be here')
            else:
                assert_allclose(chan.nb_tx * P_msg / P_noise, corrected_SNR_lin, atol=0.2,
                                err_msg='Wrong SNR')

        fading_param = zeros((nb_rx, nb_tx), complex), identity(nb_tx), identity(nb_rx)
        chan = MIMOFlatChannel(nb_tx, nb_rx, fading_param=fading_param)
        for mod in self.all_mods:
            chan.noise_std = 0
            check_noise(mod, chan, inf)
            chan.set_SNR_lin(6, Es=signal_power(mod))
            check_noise(mod, chan, 6)
            chan.set_SNR_lin(6, .5, signal_power(mod))
            check_noise(mod, chan, 3)
            chan.set_SNR_dB(0, Es=signal_power(mod))
            check_noise(mod, chan, 1)
            chan.set_SNR_dB(0, .5, signal_power(mod))
            check_noise(mod, chan, .5)


class TestMIMOTypeCheck(MIMOTestCase):
    def __init__(self):
        super(TestMIMOTypeCheck, self).__init__()

    def do(self, nb_tx, nb_rx):
        chan = MIMOFlatChannel(nb_tx, nb_rx, 0)
        with assert_raises(TypeError):
            chan.propagate(array((1, 1j)))


class TestMIMOShapes(MIMOTestCase):
    def __init__(self):
        super(TestMIMOShapes, self).__init__()

    def do(self, nb_tx, nb_rx):
        # Without padding
        chan = MIMOFlatChannel(nb_tx, nb_rx, 0)
        out = chan.propagate(ones(nb_tx * 2))
        assert_array_equal(chan.channel_gains.shape, (2, nb_rx, nb_tx),
                           err_msg='Wrong channel shape without padding')
        assert_array_equal(chan.noises.shape, (2, nb_rx),
                           err_msg='Wrong channel shape without padding')
        assert_array_equal(chan.unnoisy_output.shape, (2, nb_rx),
                           err_msg='Wrong channel shape without padding')
        assert_array_equal(out.shape, (2, nb_rx),
                           err_msg='Wrong channel shape without padding')

        # With padding
        chan = MIMOFlatChannel(nb_tx, nb_rx, 0)
        out = chan.propagate(ones(nb_tx * 2 + 1))
        assert_array_equal(chan.channel_gains.shape, (3, nb_rx, nb_tx),
                           err_msg='Wrong channel shape with padding')
        assert_array_equal(chan.noises.shape, (3, nb_rx),
                           err_msg='Wrong channel shape with padding')
        assert_array_equal(chan.unnoisy_output.shape, (3, nb_rx),
                           err_msg='Wrong channel shape with padding')
        assert_array_equal(out.shape, (3, nb_rx),
                           err_msg='Wrong channel shape with padding')


class TestMIMOkFactor(MIMOTestCase):
    def __init__(self):
        super(TestMIMOkFactor, self).__init__()

    def do(self, nb_tx, nb_rx):
        # Set seed
        seed(17121996)

        prod_nb = nb_tx * nb_rx

        # Real channel
        chan = MIMOFlatChannel(nb_tx, nb_rx)
        assert_allclose(chan.k_factor, 0,
                        err_msg='k-factor should be 0 with Rayleigh fading in SISO channels')
        mean = randn(nb_rx, nb_tx)
        mean *= sqrt(prod_nb * 0.75 / einsum('ij,ij->', absolute(mean), absolute(mean)))
        Rs = self.random_SDP_matrix(nb_tx) * sqrt(prod_nb) * 0.5
        Rr = self.random_SDP_matrix(nb_rx) * sqrt(prod_nb) * 0.5
        chan.fading_param = mean, Rs, Rr
        assert_allclose(chan.k_factor, 3,
                        err_msg='Wrong k-factor with rician fading in SISO channels')

        # Complex channel
        chan.fading_param = (zeros((nb_rx, nb_tx), complex), identity(nb_tx), identity(nb_rx))
        assert_allclose(chan.k_factor, 0,
                        err_msg='k-factor should be 0 with Rayleigh fading in SISO channels')
        mean = randn(nb_rx, nb_tx) + 1j * randn(nb_rx, nb_tx)
        mean *= sqrt(prod_nb * 0.75 / einsum('ij,ij->', absolute(mean), absolute(mean)))
        Rs = self.random_SDP_matrix(nb_tx) * sqrt(prod_nb) * 0.5
        Rr = self.random_SDP_matrix(nb_rx) * sqrt(prod_nb) * 0.5
        chan.fading_param = (mean, Rs, Rr)
        assert_allclose(chan.k_factor, 3,
                        err_msg='Wrong k-factor with rician fading in SISO channels')


if __name__ == "__main__":
    run_module_suite()
