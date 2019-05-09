# Authors: Veeresh Taranalli <veeresht@gmail.com> & Bastien Trotobas <bastien.trotobas@gmail.com>
# License: BSD 3-Clause

"""
============================================
Utilities (:mod:`commpy.utilities`)
============================================

.. autosummary::
   :toctree: generated/

   dec2bitarray         -- Integer to binary (bit array).
   bitarray2dec         -- Binary (bit array) to integer.
   hamming_dist         -- Hamming distance.
   euclid_dist          -- Squared Euclidean distance.
   upsample             -- Upsample by an integral factor (zero insertion).
   signal_power         -- Compute the power of a discrete time signal.
   link_performance     -- Estimate the BER performance of a link model with Monte Carlo simulation.

"""
from __future__ import division  # Python 2 compatibility

import numpy as np

from commpy.channels import MIMOFlatChannel

__all__ = ['dec2bitarray', 'bitarray2dec', 'hamming_dist', 'euclid_dist', 'upsample',
           'signal_power', 'link_performance']


def dec2bitarray(in_number, bit_width):
    """
    Converts a positive integer to NumPy array of the specified size containing
    bits (0 and 1).

    Parameters
    ----------
    in_number : int
        Positive integer to be converted to a bit array.

    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D ndarray of ints
        Array containing the binary representation of the input decimal.

    """

    binary_string = bin(in_number)
    length = len(binary_string)
    bitarray = np.zeros(bit_width, 'int')
    for i in range(length - 2):
        bitarray[bit_width - i - 1] = int(binary_string[length - i - 1])

    return bitarray


def bitarray2dec(in_bitarray):
    """
    Converts an input NumPy array of bits (0 and 1) to a decimal integer.

    Parameters
    ----------
    in_bitarray : 1D ndarray of ints
        Input NumPy array of bits.

    Returns
    -------
    number : int
        Integer representation of input bit array.
    """

    number = 0

    for i in range(len(in_bitarray)):
        number = number + in_bitarray[i] * pow(2, len(in_bitarray) - 1 - i)

    return number


def hamming_dist(in_bitarray_1, in_bitarray_2):
    """
    Computes the Hamming distance between two NumPy arrays of bits (0 and 1).

    Parameters
    ----------
    in_bit_array_1 : 1D ndarray of ints
        NumPy array of bits.

    in_bit_array_2 : 1D ndarray of ints
        NumPy array of bits.

    Returns
    -------
    distance : int
        Hamming distance between input bit arrays.
    """

    distance = np.bitwise_xor(in_bitarray_1, in_bitarray_2).sum()

    return distance


def euclid_dist(in_array1, in_array2):
    """
    Computes the squared euclidean distance between two NumPy arrays

    Parameters
    ----------
    in_array1 : 1D ndarray of floats
        NumPy array of real values.

    in_array2 : 1D ndarray of floats
        NumPy array of real values.

    Returns
    -------
    distance : float
        Squared Euclidean distance between two input arrays.
    """
    distance = ((in_array1 - in_array2) * (in_array1 - in_array2)).sum()

    return distance


def upsample(x, n):
    """
    Upsample the input array by a factor of n

    Adds n-1 zeros between consecutive samples of x

    Parameters
    ----------
    x : 1D ndarray
        Input array.

    n : int
        Upsampling factor

    Returns
    -------
    y : 1D ndarray
        Output upsampled array.
    """
    y = np.empty(len(x) * n, dtype=complex)
    y[0::n] = x
    zero_array = np.zeros(len(x), dtype=complex)
    for i in range(1, n):
        y[i::n] = zero_array

    return y


def signal_power(signal):
    """
    Compute the power of a discrete time signal.

    Parameters
    ----------
    signal : 1D ndarray
             Input signal.

    Returns
    -------
    P : float
        Power of the input signal.
    """

    @np.vectorize
    def square_abs(s):
        return abs(s) ** 2

    P = np.mean(square_abs(signal))
    return P


def link_performance(modem, channel, detector, SNRs, send_max, err_min, send_chunck=None, code_rate=1):
    """
    Estimate the BER performance of a link model with Monte Carlo simulation.

    Parameters
    ----------
    modem : modem object
            Modem used to modulate and demodulate.

    channel : channel object
              Channel through which the message is propagated.

    detector : function with prototype detector(y, h, constellation) or None
               Detector to decode channel output. See detectors in commpy.modulation for details on the prototype.

    SNRs : 1D arraylike
           Signal to Noise ratio in dB defined as :math:`SNR_{dB} = (E_b/N_0)_{dB} + 10 \log_{10}(R_cM_c)`
           where :math:`Rc` is the code rate and :math:`Mc` the modulation rate.

    send_max : int
               Maximum number of bits send for each SNR.

    err_min : int
              link_performance send bits until it reach err_min errors (see also send_max).

    send_chunck : int
                  Number of bits to be send at each iteration.
                  *Default*: send_chunck = err_min

    code_rate : float in (0,1]
                Rate of the used code.
                *Default*: 1 i.e. no code.
    """

    # Initialization
    BERs = np.empty_like(SNRs, dtype=float)

    # Handles the case detector is None
    if detector is None:
        def detector(y, H, constellation):
            return y

    # Set chunck size and round it to be a multiple of num_bits_symbol*nb_tx to avoid padding
    if send_chunck is None:
        send_chunck = err_min
    divider = modem.num_bits_symbol * channel.nb_tx
    send_chunck = max(divider, send_chunck // divider * divider)

    # Computations
    for id_SNR in range(len(SNRs)):
        channel.set_SNR_dB(SNRs[id_SNR], code_rate, modem.Es)
        bit_send = 0
        bit_err = 0
        while bit_send < send_max and bit_err < err_min:
            # Propagate some bits
            msg = np.random.choice((0, 1), send_chunck)
            symbs = modem.modulate(msg)
            channel_output = channel.propagate(symbs)

            # Deals with MIMO channel
            if isinstance(channel, MIMOFlatChannel):
                nb_symb_vector = len(channel_output)
                detected_msg = np.empty(nb_symb_vector * channel.nb_tx, dtype=channel_output.dtype)
                for i in range(nb_symb_vector):
                    detected_msg[channel.nb_tx * i:channel.nb_tx * (i+1)] = \
                        detector(channel_output[i], channel.channel_gains[i], modem.constellation)
            else:
                detected_msg = channel_output

            # Count errors
            received_msg = modem.demodulate(detected_msg, 'hard')
            bit_err += (msg != received_msg[:len(msg)]).sum()  # Remove MIMO padding
            bit_send += send_chunck
        BERs[id_SNR] = bit_err / bit_send
    return BERs
