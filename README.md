

[![Build Status](https://secure.travis-ci.org/veeresht/CommPy.svg?branch=master)](https://secure.travis-ci.org/veeresht/CommPy)
[![Coverage](https://coveralls.io/repos/veeresht/CommPy/badge.svg)](https://coveralls.io/r/veeresht/CommPy)
[![PyPi](https://badge.fury.io/py/scikit-commpy.svg)](https://badge.fury.io/py/scikit-commpy)
[![Docs](https://readthedocs.org/projects/commpy/badge/?version=latest)](http://commpy.readthedocs.io/en/latest/?badge=latest)

CommPy
======

CommPy is an open source toolkit implementing digital communications algorithms
in Python using NumPy and SciPy.

Objectives
----------
- To provide readable and useable implementations of algorithms used in the research, design and implementation of digital communication systems.

Available Features
------------------
Channel Coding
--------------
- Encoder for Convolutional Codes (Polynomial, Recursive Systematic). Supports all rates and puncture matrices.
- Viterbi Decoder for Convolutional Codes (Hard Decision Output).
- MAP Decoder for Convolutional Codes (Based on the BCJR algorithm).
- Encoder for a rate-1/3 systematic parallel concatenated Turbo Code.
- Turbo Decoder for a rate-1/3 systematic parallel concatenated turbo code (Based on the MAP decoder/BCJR algorithm).
- Binary Galois Field GF(2^m) with minimal polynomials and cyclotomic cosets.
- Create all possible generator polynomials for a (n,k) cyclic code.
- Random Interleavers and De-interleavers.
- Belief Propagation (BP) Decoder for LDPC Codes.

Channel Models
--------------
- Binary Erasure Channel (BEC)
- Binary Symmetric Channel (BSC)
- Binary AWGN Channel (BAWGNC)

Filters
-------
- Rectangular
- Raised Cosine (RC), Root Raised Cosine (RRC)
- Gaussian

Impairments
-----------
- Carrier Frequency Offset (CFO)

Modulation/Demodulation
-----------------------
- Phase Shift Keying (PSK)
- Quadrature Amplitude Modulation (QAM)
- OFDM Tx/Rx signal processing

Sequences
---------
- PN Sequence
- Zadoff-Chu (ZC) Sequence

Utilities
---------
- Decimal to bit-array, bit-array to decimal.
- Hamming distance, Euclidean distance.
- Upsample

FAQs
----
Why are you developing this?
----------------------------
During my coursework in communication theory and systems at UCSD, I realized that the best way to actually learn and understand the theory is to try and implement ''the Math'' in practice :). Having used Scipy before, I thought there should be a similar package for Digital Communications in Python. This is a start!

What programming languages do you use?
--------------------------------------
CommPy uses Python as its base programming language and python packages like NumPy, SciPy and Matplotlib.

How can I contribute?
---------------------
Implement any feature you want and send me a pull request :). If you want to suggest new features or discuss anything related to CommPy, please get in touch with me (veeresht@gmail.com).

How do I use CommPy?
--------------------
Requirements/Dependencies
-------------------------
- python 2.7 or above
- numpy 1.10 or above
- scipy 0.15 or above
- matplotlib 1.4 or above
- nose 1.3 or above

Installation
------------

- To use the released version on PyPi, use pip or conda to install as follows::
```
$ pip install scikit-commpy
$ conda install -c https://conda.binstar.org/veeresht scikit-commpy
```
- To work with the development branch, clone from github and install as follows::
```
$ git clone https://github.com/veeresht/CommPy.git
$ cd CommPy
$ python setup.py install
```

Citing CommPy
-------------
If you use CommPy for a publication, presentation or a demo, I request you to please cite CommPy as follows:

Veeresh Taranalli, "CommPy: Digital Communication with Python, version 0.3.0. Available at https://github.com/veeresht/CommPy", 2015.

I would also greatly appreciate your feedback if you have found CommPy useful. Just send me a mail: veeresht@gmail.com

For more details on CommPy, please visit http://veeresht.github.com/CommPy
