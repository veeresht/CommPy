import math

import numpy as np

import commpy.channelcoding.convcode as cc
import commpy.links as lk
import commpy.modulation as mod
from commpy.channels import _FlatChannel


# =============================================================================
# Convolutional Code
# =============================================================================


class Wifi80211:
    """
    This class aims to simulate the transmissions and receiving parameters of physical layer 802.11 (currently till VHT (ac))

    First the chunk is coded according to the generator matrix from the standard, having a rate of 1/2.
    Then, depending on the Modulation Coding Scheme (MCS) used, puncturing is applied to achieve other coding rates.
    For more details of which MCS map to which modulation and each coding the standard is *the* recommended place,
    but for a lighter and faster source to check  https://mcsindex.com is a good place.
    Finally the bits are then mapped to the modulation scheme in conformity to the MCS (BPSK, QPSK, 16-QAM, 64-QAM, 256-QAM).

    On the receiving the inverse operations are perform, with depuncture when MCS needs it.

    """
    # Build memory and generator matrix
    # Number of delay elements in the convolutional encoder
    # "The encoder uses a 6-stage shift register."
    # (https://pdfs.semanticscholar.org/c63b/71e43dc23b17ca57267f3b769224c64d5e33.pdf p.19)
    memory = np.array(6, ndmin=1)
    generator_matrix = np.array((133, 171), ndmin=2)  # from 802.11 standard, page 2295

    def get_modem(self):
        qpsks = [
            2,
            4,
            4,
            16,
            64,
            64,
            64,
            256,
            256
        ]
        if self.mcs == 0:
            # BPSK
            return mod.PSKModem(2)
        else:
            # Modem : QPSK
            return mod.QAMModem(qpsks[self.mcs])

    @staticmethod
    def get_puncture_matrix(numerator, denominator):
        if numerator == 1 and denominator == 2:
            return None
        # from the standard 802.11 2016
        if numerator == 2 and denominator == 3:
            # page 2297
            return [1, 1, 1, 0]
        if numerator == 3 and denominator == 4:
            # page 2297
            return [1, 1, 1, 0, 0, 1]
        if numerator == 5 and denominator == 6:
            # page 2378
            return [1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        return None

    def get_coding(self):
        coding = [
            (1, 2),
            (1, 2),
            (3, 4),
            (1, 2),
            (3, 4),
            (2, 3),
            (3, 4),
            (5, 6),
            (3, 4),
            (5, 6),
        ]
        return coding[self.mcs]

    @staticmethod
    def get_trellis():
        return cc.Trellis(Wifi80211.memory, Wifi80211.generator_matrix)

    def __init__(self, mcs):
        self.mcs = mcs
        self.modem = None

    def link_performance(self, channels: _FlatChannel, SNRs, tx_max, err_min, send_chunk=None,
                         frame_aggregation=1, receiver=None, stop_on_surpass_error=True):
        trellis1 = Wifi80211.get_trellis()
        coding = self.get_coding()
        modem = self.get_modem()

        def modulate(bits):
            res = cc.conv_encode(bits, trellis1, 'cont')
            puncture_matrix = Wifi80211.get_puncture_matrix(coding[0], coding[1])
            res_p = res
            if puncture_matrix:
                res_p = cc.puncturing(res, puncture_matrix)

            return modem.modulate(res_p)

        # Receiver function (no process required as there are no fading)
        def _receiver(y, h, constellation, noise_var):
            return modem.demodulate(y, 'soft', noise_var)

        if not receiver:
            receiver = _receiver

        # Decoder function
        def decoder_soft(msg):
            msg_d = msg
            puncture_matrix = Wifi80211.get_puncture_matrix(coding[0], coding[1])
            if puncture_matrix:
                try:
                    msg_d = cc.depuncturing(msg, puncture_matrix, math.ceil(len(msg) * coding[0] / coding[1] * 2))
                except IndexError as e:
                    print(e)
                    print("Decoded message size %d" % (math.ceil(len(msg) * coding[0] / coding[1] * 2)))
                    print("Encoded message size %d" % len(msg))
                    print("Coding %d/%d" % (coding[0], coding[1]))
            return cc.viterbi_decode(msg_d, trellis1, decoding_type='soft')

        self.model = lk.LinkModel(modulate, channels, receiver,
                                  modem.num_bits_symbol, modem.constellation, modem.Es,
                                  decoder_soft, coding[0] / coding[1])
        return self.model.link_performance(SNRs, tx_max,
                                           err_min=err_min, send_chunk=send_chunk,
                                           code_rate=coding[0] / coding[1],
                                           number_chunks_per_send=frame_aggregation,
                                           stop_on_surpass_error=stop_on_surpass_error
                                           )
