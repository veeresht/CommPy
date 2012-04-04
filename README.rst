CommPy
======

CommPy is an open source toolkit implementing digital communications algorithms 
in Python using SciPy, NumPy and Cython.

Objectives
----------
- To provide readable and useable implementations of algorithms used in the research, design and implementation of digital communication systems.

Planned Features (0.1 release)
------------------------------
- Channel Coding
	- Convolutional Codes
	- Turbo Codes
	- Low Density Parity Check (LDPC) Codes

FAQs
----
Why are you developing this?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
I am currently enrolled in graduate courses on Digital Communications and Probabilistic Coding. During the coursework, I realized that the best way to actually learn and understand the theory is to try and implement ''the Math'' in practice :). Having used Scipy before, I thought there should be a similar package for Digital Communications in Python. This is a start!

What programming languages do you use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CommPy uses Python as its base programming language and python packages like NumPy, SciPy and Matplotlib. Some algorithms which are too slow in a pure Python implementation are implemented using Cython.

How can I contribute?
~~~~~~~~~~~~~~~~~~~~~
I have put a board on Trello to track the progress of CommPy. Take a look here_. Select your feature, implement it and send me a pull request :). If you want to suggest new features or discuss anything related to CommPy, please get in touch with me.

How do I use CommPy?
~~~~~~~~~~~~~~~~~~~~
Requirements
^^^^^^^^^^^^
- Python 2.7 or above
- NumPy 1.6 or above
- SciPy 0.10 or above
- Matplotlib 1.1 or above
- Cython 0.15 or above

Currently there is no installer for CommPy. You can clone the github repo and starting using it. Remember to build the Cython modules. You can also look at a very preliminary version of the Commpy Documentation_.

.. _here: https://trello.com/board/commpy/4f44785f28107d10684bbd7d 
.. _Documentation: http://veeresht.github.com/CommPy/commpydoc/commpydoc.html
