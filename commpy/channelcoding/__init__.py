"""
============================================
Channel Coding (:mod:`commpy.channelcoding`)
============================================

.. module:: commpy.channelcoding

Convolutional Codes
===================

.. autosummary::
    :toctree: generated/
    
    Trellis          -- Class representing convolutional code trellis.
    conv_encode      -- Convolutional Encoder.
    viterbi_decode   -- Convolutional Decoder using the Viterbi algorithm.

Interleavers and De-interleavers
================================

.. autosummary::
    :toctree: generated/

    rand_interlv    -- Random Interleaver.
    rand_deinterlv  -- Random De-interleaver.

"""

from convcode import Trellis, conv_encode, viterbi_decode
from interleavers import *
from turbo import turbo_encode, map_decode, turbo_decode

