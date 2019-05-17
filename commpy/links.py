# Authors: Youness Akourim <akourim97@gmail.com> & Bastien Trotobas <bastien.trotobas@gmail.com>
# License: BSD 3-Clause

"""
============================================
Links (:mod:`commpy.links`)
============================================

.. autosummary::
   :toctree: generated/

   link_performance     -- Estimate the BER performance of a link model with Monte Carlo simulation.
   linkModel            -- Link model object.
"""
from __future__ import division  # Python 2 compatibility

import numpy as np
from commpy.channels import MIMOFlatChannel

__all__ = ['link_performance', 'linkModel']


def link_performance(link_model, SNRs, send_max, err_min, send_chunk=None, code_rate=1):
    """
    Estimate the BER performance of a link model with Monte Carlo simulation.

    Parameters
    ----------
    link_model : linkModel object.

    SNRs : 1D arraylike
           Signal to Noise ratio in dB defined as :math:`SNR_{dB} = (E_b/N_0)_{dB} + 10 \log_{10}(R_cM_c)`
           where :math:`Rc` is the code rate and :math:`Mc` the modulation rate.

    send_max : int
               Maximum number of bits send for each SNR.

    err_min : int
              link_performance send bits until it reach err_min errors (see also send_max).

    send_chunk : int
                  Number of bits to be send at each iteration.
                  *Default*: send_chunck = err_min

    code_rate : float in (0,1]
                Rate of the used code.
                *Default*: 1 i.e. no code.

    Returns
    -------
    BERs : 1d ndarray
           Estimated Bit Error Ratio corresponding to each SNRs
    """

    # Initialization
    BERs = np.empty_like(SNRs, dtype=float)
    # Set chunk size and round it to be a multiple of num_bits_symbol*nb_tx to avoid padding
    if send_chunk is None:
        send_chunk = err_min
    divider = link_model.num_bits_symbol * link_model.channel.nb_tx
    send_chunk = max(divider, send_chunk // divider * divider)

    # Computations
    for id_SNR in range(len(SNRs)):
        link_model.channel.set_SNR_dB(SNRs[id_SNR], code_rate, link_model.Es)
        bit_send = 0
        bit_err = 0
        while bit_send < send_max and bit_err < err_min:
            # Propagate some bits
            msg = np.random.choice((0, 1), send_chunk)
            symbs = link_model.modulate(msg)
            channel_output = link_model.channel.propagate(symbs)

            # Deals with MIMO channel
            if isinstance(link_model.channel, MIMOFlatChannel):
                nb_symb_vector = len(channel_output)
                received_msg = np.empty(nb_symb_vector * link_model.channel.nb_tx, dtype=channel_output.dtype)
                for i in range(nb_symb_vector):
                     received_msg[link_model.channel.nb_tx * i:link_model.channel.nb_tx * (i+1)] = \
                         link_model.receive(channel_output[i], link_model.channel.channel_gains[i], link_model.constellation)
            else:
                received_msg = channel_output
            # Count errors
            bit_err += (msg != received_msg[:len(msg)]).sum()  # Remove MIMO padding
            bit_send += send_chunk
        BERs[id_SNR] = bit_err / bit_send
    return BERs


class linkModel:
    """
        Construct a link model.

        Parameters
        ----------
        modulate : function with same prototype as Modem.modulate

        channel : _FlatChannel object

        receive : function with prototype receive(y, H, constellation) that return a binary array.
                    y : 1D ndarray of floats
                        Received complex symbols (shape: num_receive_antennas x 1)

                    h : 2D ndarray of floats
                        Channel Matrix (shape: num_receive_antennas x num_transmit_antennas)

                    constellation : 1D ndarray of floats

        num_bits_symbols : int

        constellation : array of float or complex

        Es : float
             Average energy per symbols.
             *Default* Es=1.

        Attributes
        ----------
        modulate : function with same prototype as Modem.modulate

        channel : _FlatChannel object

        receive : function with prototype receive(y, H, constellation) that return a binary array.
                    y : 1D ndarray of floats
                        Received complex symbols (shape: num_receive_antennas x 1)

                    h : 2D ndarray of floats
                        Channel Matrix (shape: num_receive_antennas x num_transmit_antennas)

                    constellation : 1D ndarray of floats

        num_bits_symbols : int

        constellation : array of float or complex

        Es : float
             Average energy per symbols.
             *Default* Es=1.
        """
    def __init__(self, modulate, channel, receive, num_bits_symbol, constellation, Es=1):
        self.modulate = modulate
        self.channel = channel
        self.receive = receive
        self.num_bits_symbol = num_bits_symbol
        self.constellation = constellation
        self.Es = Es
