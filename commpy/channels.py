# Authors: Veeresh Taranalli <veeresht@gmail.com> & Bastien Trotobas <bastien.trotobas@gmail.com>
# License: BSD 3-Clause

"""
============================================
Channel Models (:mod:`commpy.channels`)
============================================

.. autosummary::
   :toctree: generated/

   SISOFlatChannel     -- SISO Channel with Rayleigh or Rician fading.
   MIMOFlatChannel     -- MIMO Channel with Rayleigh or Rician fading.
   bec                 -- Binary Erasure Channel.
   bsc                 -- Binary Symmetric Channel.
   awgn                -- Additive White Gaussian Noise Channel.

"""

from __future__ import division, print_function  # Python 2 compatibility

from numpy import complex, abs, sqrt, sum, zeros, identity, hstack, einsum, trace, kron, absolute, fromiter
from numpy.random import randn, random, standard_normal
from scipy.linalg import sqrtm

__all__ = ['SISOFlatChannel', 'MIMOFlatChannel', 'bec', 'bsc', 'awgn']


class _FlatChannel(object):

    def __init__(self):
        self.noises = None
        self.channel_gains = None
        self.unnoisy_output = None

    def generate_noises(self, dims):

        """
        Generates the white gaussian noise with the right standard deviation and saves it.

        Parameters
        ----------
        dims : int or tuple of ints
                Shape of the generated noise.
        """

        # Check channel state
        assert self.noise_std is not None, "Noise standard deviation must be set before propagation."

        # Generate noises
        if self.isComplex:
            self.noises = (standard_normal(dims) + 1j * standard_normal(dims)) * self.noise_std * 0.5
        else:
            self.noises = standard_normal(dims) * self.noise_std

    def set_SNR_dB(self, SNR_dB, code_rate=1, Es=1):

        """
        Sets the the noise standard deviation based on SNR expressed in dB.

        Parameters
        ----------
        SNR_dB      : float
                        Signal to Noise Ratio expressed in dB.

        code_rate   : float in (0,1]
                        Rate of the used code.

        Es          : positive float
                        Average symbol energy
        """

        self.noise_std = sqrt((self.isComplex + 1) * self.nb_tx * Es / (code_rate * 10 ** (SNR_dB / 10)))

    def set_SNR_lin(self, SNR_lin, code_rate=1, Es=1):

        """
        Sets the the noise standard deviation based on SNR expressed in its linear form.

        Parameters
        ----------
        SNR_lin     : float
                        Signal to Noise Ratio as a linear ratio.

        code_rate   : float in (0,1]
                        Rate of the used code.

        Es          : positive float
                        Average symbol energy
        """

        self.noise_std = sqrt((self.isComplex + 1) * self.nb_tx * Es / (code_rate * SNR_lin))

    @property
    def isComplex(self):
        """ Read-only - True if the channel is complex, False if not."""
        return self._isComplex


class SISOFlatChannel(_FlatChannel):
    """
    Constructs a SISO channel with a flat fading.
    The channel coefficient are normalized i.e. the mean magnitude is 1.

    Parameters
    ----------
    noise_std    : float, optional
                   Noise standard deviation.
                   *Default* value is None and then the value must set later.

    fading_param : tuple of 2 floats, optional
                   Parameters of the fading (see attribute for details).
                   *Default* value is (1,0) i.e. no fading.

    Attributes
    ----------
    fading_param : tuple of 2 floats
                   Parameters of the fading. The complete tuple must be set each time.
                   Raise ValueError when sets with value that would lead to a non-normalized channel.

                        * fading_param[0] refers to the mean of the channel gain (Line Of Sight component).

                        * fading_param[1] refers to the variance of the channel gain (Non Line Of Sight component).

                   Classical fadings:

                        * (1, 0): no fading.

                        * (0, 1): Rayleigh fading.

                        * Others: rician fading.

    noise_std       : float
                       Noise standard deviation. None is the value has not been set yet.

    isComplex       : Boolean, Read-only
                        True if the channel is complex, False if not.
                        The value is set together with fading_param based on the type of fading_param[0].

    k_factor        : positive float, Read-only
                        Fading k-factor, the power ratio between LOS and NLOS.

    nb_tx           : int = 1, Read-only
                        Number of Tx antennas.

    nb_rx           : int = 1, Read-only
                        Number of Rx antennas.

    noises          : 1D ndarray
                        Last noise generated. None if no noise has been generated yet.

    channel_gains   : 1D ndarray
                        Last channels gains generated. None if no channels has been generated yet.

    unnoisy_output  : 1D ndarray
                        Last transmitted message without noise. None if no message has been propagated yet.

    Raises
    ------
    ValueError
                    If the fading parameters would lead to a non-normalized channel.
                    The condition is :math:`|param[1]| + |param[0]|^2 = 1`
    """

    @property
    def nb_tx(self):
        """ Read-only - Number of Tx antennas, set to 1 for SISO channel."""
        return 1

    @property
    def nb_rx(self):
        """ Read-only - Number of Rx antennas, set to 1 for SISO channel."""
        return 1

    def __init__(self, noise_std=None, fading_param=(1, 0)):
        super(SISOFlatChannel, self).__init__()
        self.noise_std = noise_std
        self.fading_param = fading_param

    def propagate(self, msg):

        """
        Propagates a message through the channel.

        Parameters
        ----------
        msg : 1D ndarray
                Message to propagate.

        Returns
        -------
        channel_output : 1D ndarray
                            Message after application of the fading and addition of noise.

        Raises
        ------
        TypeError
                        If the input message is complex but the channel is real.

        AssertionError
                        If the noise standard deviation as not been set yet.
        """

        if isinstance(msg[0], complex) and not self.isComplex:
            raise TypeError('Trying to propagate a complex message in a real channel.')
        nb_symb = len(msg)

        # Generate noise
        self.generate_noises(nb_symb)

        # Generate channel
        self.channel_gains = self.fading_param[0]
        if self.isComplex:
            self.channel_gains += (standard_normal(nb_symb) + 1j * standard_normal(nb_symb)) * sqrt(0.5 * self.fading_param[1])
        else:
            self.channel_gains += standard_normal(nb_symb) * sqrt(self.fading_param[1])

        # Generate outputs
        self.unnoisy_output = self.channel_gains * msg
        return self.unnoisy_output + self.noises

    @property
    def fading_param(self):
        """ Parameters of the fading (see class attribute for details). """
        return self._fading_param

    @fading_param.setter
    def fading_param(self, fading_param):
        if fading_param[1] + absolute(fading_param[0]) ** 2 != 1:
            raise ValueError("With this parameters, the channel would add or remove energy.")

        self._fading_param = fading_param
        self._isComplex = isinstance(fading_param[0], complex)

    @property
    def k_factor(self):
        """ Read-only - Fading k-factor, the power ratio between LOS and NLOS """
        return absolute(self.fading_param[0]) ** 2 / absolute(self.fading_param[1])


class MIMOFlatChannel(_FlatChannel):
    """
    Constructs a MIMO channel with a flat fading based on the Kronecker model.
    The channel coefficient are normalized i.e. the mean magnitude is 1.

    Parameters
    ----------
    nb_tx        : int >= 1
                   Number of Tx antennas.

    nb_rx        : int >= 1
                   Number of Rx antennas.

    noise_std    : float, optional
                   Noise standard deviation.
                   *Default* value is None and then the value must set later.

    fading_param : tuple of 3 floats, optional
                   Parameters of the fading. The complete tuple must be set each time.
                   *Default* value is (zeros((nb_rx, nb_tx)), identity(nb_tx), identity(nb_rx)) i.e. Rayleigh fading.

    Attributes
    ----------
    fading_param : tuple of 3 2D ndarray
                   Parameters of the fading.
                   Raise ValueError when sets with value that would lead to a non-normalized channel.

                        * fading_param[0] refers to the mean of the channel gain (Line Of Sight component).

                        * fading_param[1] refers to the transmit-side spatial correlation matrix of the channel.

                        * fading_param[2] refers to the receive-side spatial correlation matrix of the channel.

                   Classical fadings:

                        * (zeros((nb_rx, nb_tx)), identity(nb_tx), identity(nb_rx)): Uncorrelated Rayleigh fading.

    noise_std       : float
                       Noise standard deviation. None is the value has not been set yet.

    isComplex       : Boolean, Read-only
                        True if the channel is complex, False if not.
                        The value is set together with fading_param based on the type of fading_param[0].

    k_factor        : positive float, Read-only
                        Fading k-factor, the power ratio between LOS and NLOS.

    nb_tx           : int
                        Number of Tx antennas.

    nb_rx           : int
                        Number of Rx antennas.

    noises          : 2D ndarray
                        Last noise generated. None if no noise has been generated yet.
                        noises[i] is the noise vector of size nb_rx for the i-th message vector.

    channel_gains   : 2D ndarray
                        Last channels gains generated. None if no channels has been generated yet.
                        channel_gains[i] is the channel matrix of size (nb_rx x nb_tx) for the i-th message vector.

    unnoisy_output  : 1D ndarray
                        Last transmitted message without noise. None if no message has been propageted yet.
                        unnoisy_output[i] is the transmitted message without noise of size nb_rx for the i-th message vector.

    Raises
    ------
    ValueError
                    If the fading parameters would lead to a non-normalized channel.
                    The condition is :math:`NLOS + LOS = nb_{tx} * nb_{rx}` where

                        * :math:`NLOS = tr(param[1]^T \otimes param[2])`

                        * :math:`LOS = \sum|param[0]|^2`
    """

    def __init__(self, nb_tx, nb_rx, noise_std=None, fading_param=None):
        super(MIMOFlatChannel, self).__init__()
        self.nb_tx = nb_tx
        self.nb_rx = nb_rx
        self.noise_std = noise_std

        if fading_param is None:
            self.fading_param = (zeros((nb_rx, nb_tx)), identity(nb_tx), identity(nb_rx))
        else:
            self.fading_param = fading_param

    def propagate(self, msg):

        """
        Propagates a message through the channel.

        Parameters
        ----------
        msg : 1D ndarray
                Message to propagate.

        Returns
        -------
        channel_output : 2D ndarray
                         Message after application of the fading and addition of noise.
                         channel_output[i] is th i-th received symbol of size nb_rx.

        Raises
        ------
        TypeError
                        If the input message is complex but the channel is real.

        AssertionError
                        If the noise standard deviation noise_std as not been set yet.
        """

        if isinstance(msg[0], complex) and not self.isComplex:
            raise TypeError('Trying to propagate a complex message in a real channel.')
        (nb_vect, mod) = divmod(len(msg), self.nb_tx)

        # Add padding if required
        if mod:
            msg = hstack((msg, zeros(self.nb_tx - mod)))
            nb_vect += 1

        # Reshape msg as vectors sent on each antennas
        msg = msg.reshape(nb_vect, -1)

        # Generate noises
        self.generate_noises((nb_vect, self.nb_rx))

        # Generate channel uncorrelated channel
        dims = (nb_vect, self.nb_rx, self.nb_tx)
        if self.isComplex:
            self.channel_gains = (standard_normal(dims) + 1j * standard_normal(dims)) * sqrt(0.5)
        else:
            self.channel_gains = standard_normal(dims)

        # Add correlation and mean
        einsum('ij,ajk,lk->ail', sqrtm(self.fading_param[2]), self.channel_gains, sqrtm(self.fading_param[1]),
               out=self.channel_gains, optimize='greedy')
        self.channel_gains += self.fading_param[0]

        # Generate outputs
        self.unnoisy_output = einsum('ijk,ik->ij', self.channel_gains, msg)
        return self.unnoisy_output + self.noises

    @property
    def fading_param(self):
        """ Parameters of the fading (see class attribute for details). """
        return self._fading_param

    @fading_param.setter
    def fading_param(self, fading_param):
        NLOS_gain = trace(kron(fading_param[1].T, fading_param[2]))
        LOS_gain = einsum('ij,ij->', absolute(fading_param[0]), absolute(fading_param[0]))
        if absolute(NLOS_gain + LOS_gain - self.nb_tx * self.nb_rx) > 1e-3:
            raise ValueError("With this parameters, the channel would add or remove energy.")

        self._fading_param = fading_param
        self._isComplex = isinstance(fading_param[0][0, 0], complex)

    @property
    def k_factor(self):
        """ Read-only - Fading k-factor, the power ratio between LOS and NLOS """
        NLOS_gain = trace(kron(self.fading_param[1].T, self.fading_param[2]))
        LOS_gain = einsum('ij,ij->', absolute(self.fading_param[0]), absolute(self.fading_param[0]))
        return LOS_gain / NLOS_gain

    def uncorr_rayleigh_fading(self, dtype):
        """ Set the fading parameters to an uncorrelated Rayleigh channel.

        Parameters
        ----------
        dtype : dtype
                Type of the channel
        """
        self.fading_param = zeros((self.nb_rx, self.nb_tx), dtype), identity(self.nb_tx), identity(self.nb_rx)

    def expo_corr_rayleigh_fading(self, t, r):
        """ Set the fading parameters to a complex correlated Rayleigh channel following the exponential model.

        See: S. L. Loyka, « Channel capacity of MIMO architecture using the exponential correlation matrix », IEEE
            Commun. Lett., vol. 5, nᵒ 9, p. 369‑371, sept. 2001.

        Parameters
        ----------
        t : complex with abs(t) = 1
            Correlation coefficient for the transceiver.

        r : complex with abs(r) = 1
            Correlation coefficient for the receiver.

        Raises
        ------
        ValueError
                    If abs(t) != 1 or abs(r) != 1
        """
        # Check inputs
        if abs(t) - 1 > 1e-4:
            raise ValueError('abs(t) must be one.')
        if abs(r) - 1 > 1e-4:
            raise ValueError('abs(r) must be one.')

        # Construct the exponent matrix
        expo_tx = fromiter((j - i for i in range(self.nb_tx) for j in range(self.nb_tx)), int, self.nb_tx ** 2)
        expo_rx = fromiter((j - i for i in range(self.nb_rx) for j in range(self.nb_rx)), int, self.nb_rx ** 2)

        # Reshape
        expo_tx = expo_tx.reshape(self.nb_tx, self.nb_tx)
        expo_rx = expo_rx.reshape(self.nb_rx, self.nb_rx)

        # Set fading
        self.fading_param = zeros((self.nb_rx, self.nb_tx), complex), t ** expo_tx, r ** expo_rx

    def uncorr_rician_fading(self, mean, k_factor):
        """ Set the fading parameters to an uncorrelated rician channel.

        mean will be scaled to fit the required k-factor.

        Parameters
        ----------
        mean : ndarray (shape: nb_rx x nb_tx)
               Mean of the channel gain.

        k_factor : positive float
                   Requested k-factor (the power ratio between LOS and NLOS).
        """
        nb_antennas = mean.size
        NLOS_gain = nb_antennas / (k_factor + 1)
        mean = mean * sqrt(k_factor * NLOS_gain / einsum('ij,ij->', absolute(mean), absolute(mean)))
        self.fading_param = mean, identity(self.nb_tx) * NLOS_gain / nb_antennas, identity(self.nb_rx)

    def expo_corr_rician_fading(self, mean, k_factor, t, r):
        """ Set the fading parameters to a complex correlated rician channel following the exponential model.

        See: S. L. Loyka, « Channel capacity of MIMO architecture using the exponential correlation matrix », IEEE
            Commun. Lett., vol. 5, nᵒ 9, p. 369‑371, sept. 2001.

        mean and correlation matricies will be scaled to fit the required k-factor.

        Parameters
        ----------
        mean : ndarray (shape: nb_rx x nb_tx)
               Mean of the channel gain.

        k_factor : positive float
                   Requested k-factor (the power ratio between LOS and NLOS).

        t : complex with abs(t) = 1
            Correlation coefficient for the transceiver.

        r : complex with abs(r) = 1
            Correlation coefficient for the receiver.

        Raises
        ------
        ValueError
                    If abs(t) != 1 or abs(r) != 1
        """
        # Check inputs
        if abs(t) - 1 > 1e-4:
            raise ValueError('abs(t) must be one.')
        if abs(r) - 1 > 1e-4:
            raise ValueError('abs(r) must be one.')

        # Scaling
        nb_antennas = mean.size
        NLOS_gain = nb_antennas / (k_factor + 1)
        mean = mean * sqrt(k_factor * NLOS_gain / einsum('ij,ij->', absolute(mean), absolute(mean)))

        # Construct the exponent matrix
        expo_tx = fromiter((j - i for i in range(self.nb_tx) for j in range(self.nb_tx)), int, self.nb_tx ** 2)
        expo_rx = fromiter((j - i for i in range(self.nb_rx) for j in range(self.nb_rx)), int, self.nb_rx ** 2)

        # Reshape
        expo_tx = expo_tx.reshape(self.nb_tx, self.nb_tx)
        expo_rx = expo_rx.reshape(self.nb_rx, self.nb_rx)

        # Set fading
        self.fading_param = mean, t ** expo_tx * NLOS_gain / nb_antennas, r ** expo_rx


def bec(input_bits, p_e):
    """
    Binary Erasure Channel.

    Parameters
    ----------
    input_bits : 1D ndarray containing {0, 1}
        Input arrary of bits to the channel.

    p_e : float in [0, 1]
        Erasure probability of the channel.

    Returns
    -------
    output_bits : 1D ndarray containing {0, 1}
        Output bits from the channel.
    """
    output_bits = input_bits.copy()
    output_bits[random(len(output_bits)) <= p_e] = -1
    return output_bits


def bsc(input_bits, p_t):
    """
    Binary Symmetric Channel.

    Parameters
    ----------
    input_bits : 1D ndarray containing {0, 1}
        Input arrary of bits to the channel.

    p_t : float in [0, 1]
        Transition/Error probability of the channel.

    Returns
    -------
    output_bits : 1D ndarray containing {0, 1}
        Output bits from the channel.
    """
    output_bits = input_bits.copy()
    flip_locs = (random(len(output_bits)) <= p_t)
    output_bits[flip_locs] = 1 ^ output_bits[flip_locs]
    return output_bits


# Kept for retro-compatibility. Use FlatChannel for new programs.
def awgn(input_signal, snr_dB, rate=1.0):
    """
    Addditive White Gaussian Noise (AWGN) Channel.

    Parameters
    ----------
    input_signal : 1D ndarray of floats
        Input signal to the channel.

    snr_dB : float
        Output SNR required in dB.

    rate : float
        Rate of the a FEC code used if any, otherwise 1.

    Returns
    -------
    output_signal : 1D ndarray of floats
        Output signal from the channel with the specified SNR.
    """

    avg_energy = sum(abs(input_signal) * abs(input_signal)) / len(input_signal)
    snr_linear = 10 ** (snr_dB / 10.0)
    noise_variance = avg_energy / (2 * rate * snr_linear)

    if isinstance(input_signal[0], complex):
        noise = (sqrt(noise_variance) * randn(len(input_signal))) + (sqrt(noise_variance) * randn(len(input_signal))*1j)
    else:
        noise = sqrt(2 * noise_variance) * randn(len(input_signal))

    output_signal = input_signal + noise

    return output_signal
