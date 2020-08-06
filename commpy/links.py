# Authors: CommPy contributors
# License: BSD 3-Clause

"""
============================================
Links (:mod:`commpy.links`)
============================================

.. autosummary::
   :toctree: generated/

   link_performance     -- Estimate the BER performance of a link model with Monte Carlo simulation.
   LinkModel            -- Link model object.
   idd_decoder          -- Produce the decoder function to model a MIMO IDD decoder.
"""
from __future__ import division  # Python 2 compatibility

import math
from inspect import getfullargspec

import numpy as np

from commpy.channels import MIMOFlatChannel

__all__ = ['link_performance', 'LinkModel', 'idd_decoder']


def link_performance(link_model, SNRs, send_max, err_min, send_chunk=None, code_rate=1):
    """
    Estimate the BER performance of a link model with Monte Carlo simulation.
    Equivalent to call link_model.link_performance(SNRs, send_max, err_min, send_chunk, code_rate).

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
                  Number of bits to be send at each iteration. This is also the frame length of the decoder if available
                  so it should be large enough regarding the code type.
                  *Default*: send_chunck = err_min

    code_rate : float in (0,1]
                Rate of the used code.
                *Default*: 1 i.e. no code.

    Returns
    -------
    BERs : 1d ndarray
           Estimated Bit Error Ratio corresponding to each SNRs
    """

    return link_model.link_performance(SNRs, send_max, err_min, send_chunk, code_rate)


class LinkModel:
    """
    Construct a link model.

    Parameters
    ----------
    modulate : function with same prototype as Modem.modulate

    channel : FlatChannel object

    receive : function with prototype receive(y, H, constellation, noise_var) that return a binary array.
                y : 1D ndarray
                    Received complex symbols (shape: num_receive_antennas x 1)

                h : 2D ndarray
                    Channel Matrix (shape: num_receive_antennas x num_transmit_antennas)

                constellation : 1D ndarray

                noise_var : positive float
                            Noise variance

    num_bits_symbols : int

    constellation : array of float or complex

    Es : float
         Average energy per symbols.
         *Default* Es=1.

    decoder : function with prototype decoder(array) or decoder(y, H, constellation, noise_var, array) that return a
                binary array.
              *Default* is no process.

    rate : float
        Code rate.
        *Default* is 1.

    Attributes
    ----------
    modulate : function with same prototype as Modem.modulate

    channel : _FlatChannel object

    receive : function with prototype receive(y, H, constellation, noise_var) that return a binary array.
                y : 1D ndarray
                    Received complex symbols (shape: num_receive_antennas x 1)

                h : 2D ndarray
                    Channel Matrix (shape: num_receive_antennas x num_transmit_antennas)

                constellation : 1D ndarray

                noise_var : positive float
                            Noise variance

    num_bits_symbols : int

    constellation : array of float or complex

    Es : float
         Average energy per symbols.

    decoder : function with prototype decoder(binary array) that return a binary array.
              *Default* is no process.

    rate : float
        Code rate.
        *Default* is 1.
    """

    def __init__(self, modulate, channel, receive, num_bits_symbol, constellation, Es=1, decoder=None, rate=1):
        self.modulate = modulate
        self.channel = channel
        self.receive = receive
        self.num_bits_symbol = num_bits_symbol
        self.constellation = constellation
        self.Es = Es
        self.rate = rate

        if decoder is None:
            self.decoder = lambda msg: msg
        else:
            self.decoder = decoder
        self.full_simulation_results = None

    def link_performance(self, SNRs, send_max, err_min, send_chunk=None, code_rate=1):
        """
        Estimate the BER performance of a link model with Monte Carlo simulation.

        Parameters
        ----------
        SNRs : 1D arraylike
               Signal to Noise ratio in dB defined as :math:`SNR_{dB} = (E_b/N_0)_{dB} + 10 \log_{10}(R_cM_c)`
               where :math:`Rc` is the code rate and :math:`Mc` the modulation rate.

        send_max : int
                   Maximum number of bits send for each SNR.

        err_min : int
                  link_performance send bits until it reach err_min errors (see also send_max).

        send_chunk : int
                      Number of bits to be send at each iteration. This is also the frame length of the decoder if available
                      so it should be large enough regarding the code type.
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
        BERs = np.zeros_like(SNRs, dtype=float)
        CEs = np.zeros_like(SNRs, dtype=float)  # Chunk Errors
        NCs = np.zeros_like(SNRs, dtype=float)  # Number of Chunks
        # Set chunk size and round it to be a multiple of num_bits_symbol*nb_tx to avoid padding
        if send_chunk is None:
            send_chunk = err_min
        divider = self.num_bits_symbol * self.channel.nb_tx
        send_chunk = max(divider, send_chunk // divider * divider)

        receive_size = self.channel.nb_tx * self.num_bits_symbol
        full_args_decoder = len(getfullargspec(self.decoder).args) > 1

        # Computations
        for id_SNR in range(len(SNRs)):
            self.channel.set_SNR_dB(SNRs[id_SNR], code_rate, self.Es)
            total_bit_send = 0
            total_bit_err = 0
            total_chunk_loss = 0
            total_chunk_count = 0
            while total_bit_send < send_max and total_bit_err < err_min:
                # Propagate some bits
                msg = np.random.choice((0, 1), send_chunk)
                symbs = self.modulate(msg)
                channel_output = self.channel.propagate(symbs)

                # Deals with MIMO channel
                if isinstance(self.channel, MIMOFlatChannel):
                    nb_symb_vector = len(channel_output)
                    received_msg = np.empty(int(math.ceil(len(msg) / self.rate)), dtype=np.int8)
                    for i in range(nb_symb_vector):
                        received_msg[receive_size * i:receive_size * (i + 1)] = \
                            self.receive(channel_output[i], self.channel.channel_gains[i],
                                         self.constellation, self.channel.noise_std ** 2)
                else:
                    received_msg = self.receive(channel_output, self.channel.channel_gains,
                                                self.constellation, self.channel.noise_std ** 2)
                # Count errors
                if full_args_decoder:
                    decoded_bits = self.decoder(channel_output, self.channel.channel_gains,
                                                self.constellation, self.channel.noise_std ** 2,
                                                received_msg, self.channel.nb_tx * self.num_bits_symbol)
                    bit_err = (msg != decoded_bits[:len(msg)]).sum()
                else:
                    bit_err = (msg != self.decoder(received_msg)[:len(msg)]).sum()
                total_bit_err += bit_err
                total_chunk_count += 1
                total_chunk_loss += 1 if bit_err > 0 else 0
                total_bit_send += send_chunk
            BERs[id_SNR] = total_bit_err / total_bit_send
            CEs[id_SNR] = total_chunk_loss
            NCs[id_SNR] = total_chunk_count
            if total_bit_err < err_min:
                break
        self.full_simulation_results = BERs, CEs, NCs
        return BERs


def idd_decoder(detector, decoder, decision, n_it):
    """
    Produce a decoder function that model the specified MIMO iterative detection and decoding (IDD) process.
    The returned function can be used as is to build a working LinkModel object.

    Parameters
    ----------
    detector : function with prototype detector(y, H, constellation, noise_var, a_priori) that return a LLRs array.
                y : 1D ndarray
                    Received complex symbols (shape: num_receive_antennas x 1).

                h : 2D ndarray
                    Channel Matrix (shape: num_receive_antennas x num_transmit_antennas).

                constellation : 1D ndarray.

                noise_var : positive float
                            Noise variance.

                a_priori : 1D ndarray of floats
                            A priori as Log-Likelihood Ratios.

    decoder : function with prototype(LLRs) that return a LLRs array.
            LLRs : 1D ndarray of floats
            A priori as Log-Likelihood Ratios.

    decision : function wih prototype(LLRs) that return a binary 1D-array that model the decision to extract the
        information bits from the LLRs array.

    n_it : positive integer
            Number or iteration during the IDD process.

    Returns
    -------
    decode : function useable as it is to build a LinkModel object that produce a bit array from the parameters
                y : 1D ndarray
                    Received complex symbols (shape: num_receive_antennas x 1).

                h : 2D ndarray
                    Channel Matrix (shape: num_receive_antennas x num_transmit_antennas).

                constellation : 1D ndarray

                noise_var : positive float
                            Noise variance.

                bits_per_send : positive integer
                                Number or bit send at each symbol vector.
    """

    def decode(y, h, constellation, noise_var, a_priori, bits_per_send):
        a_priori_decoder = a_priori.copy()
        nb_vect, nb_rx, nb_tx = h.shape
        for iteration in range(n_it):
            a_priori_detector = (decoder(a_priori_decoder) - a_priori_decoder)
            for i in range(nb_vect):
                a_priori_decoder[i * bits_per_send:(i + 1) * bits_per_send] = \
                    detector(y[i], h[i], constellation, noise_var,
                             a_priori_detector[i * bits_per_send:(i + 1) * bits_per_send])
            a_priori_decoder -= a_priori_detector
        return decision(a_priori_decoder + a_priori_detector)

    return decode
