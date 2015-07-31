.. CommPy documentation master file, created by
   sphinx-quickstart on Sun Jan 29 23:37:16 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CommPy
==================================

CommPy is an open source package implementing digital communications algorithms
in Python using NumPy, SciPy and Matplotlib.

Available Features
------------------
Channel Coding
~~~~~~~~~~~~~~
- Encoder for Convolutional Codes (Polynomial, Recursive Systematic). Supports all rates and puncture matrices.
- Viterbi Decoder for Convolutional Codes (Hard Decision Output).
- MAP Decoder for Convolutional Codes (Based on the BCJR algorithm).
- Encoder for a rate-1/3 systematic parallel concatenated Turbo Code.
- Turbo Decoder for a rate-1/3 systematic parallel concatenated turbo code (Based on the MAP decoder/BCJR algorithm).
- Binary Galois Field GF(2^m) with minimal polynomials and cyclotomic cosets.
- Create all possible generator polynomials for a (n,k) cyclic code.
- Random Interleavers and De-interleavers.

Channel Models
~~~~~~~~~~~~~~
- Binary Erasure Channel (BEC)
- Binary Symmetric Channel (BSC)
- Binary AWGN Channel (BAWGNC)

Filters
~~~~~~~
- Rectangular
- Raised Cosine (RC), Root Raised Cosine (RRC)
- Gaussian

Impairments
~~~~~~~~~~~
- Carrier Frequency Offset (CFO)

Modulation/Demodulation
~~~~~~~~~~~~~~~~~~~~~~~
- Phase Shift Keying (PSK)
- Quadrature Amplitude Modulation (QAM)
- OFDM Tx/Rx signal processing

Sequences
~~~~~~~~~
- PN Sequence
- Zadoff-Chu (ZC) Sequence

Utilities
~~~~~~~~~
- Decimal to bit-array, bit-array to decimal.
- Hamming distance, Euclidean distance.
- Upsample



Reference
---------
.. toctree::
    :maxdepth: 4

    channelcoding
    channels
    filters
    impairments
    modulation
    sequences
    utilities
