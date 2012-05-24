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


Turbo Codes
===========

.. autosummary::
    :toctree: generated/

    turbo_encode    -- Turbo Encoder.
    map_decode      -- Convolutional Code decoder using MAP algorithm.
    turbo_decode    -- Turbo Decoder.

Interleavers and De-interleavers
================================

.. autosummary::
    :toctree: generated/

    RandInterlv    -- Random Interleaver.

"""

from convcode import Trellis, conv_encode, viterbi_decode
from interleavers import *
from turbo import turbo_encode, map_decode, turbo_decode

