CommPy
======

CommPy is an open source toolkit implementing digital communications algorithms 
in Python using SciPy, NumPy and Cython.

What's implemented?
-------------------
- Convolutional Encoder (only polynomial) for arbitrary rates
- Viterbi Decoder (hard decision and unquantized soft decision decoding in the terminated mode)
- Pulse Shaping Filters (FIR) - Raised Cosine, Root Raised Cosine, Gaussian 

To Do
-----
- Support for recursive and recursive systematic convolutional encoding
- Quantized soft decision decoding in the Viterbi decoder
- Support truncated and continuous modes in the Viterbi decoder
- Computation of the distance spectrum of a convolutional code
