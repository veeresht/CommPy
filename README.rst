CommPy
======

CommPy is an open source toolkit implementing digital communications algorithms 
in Python using SciPy, NumPy and Cython.

Objectives
----------
- To provide readable and useable implementations of algorithms used in the research, design and implementation of digital communication systems.

FAQs
----
Why are you developing this?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
During my coursework in communication theory and systems at UCSD, I realized that the best way to actually learn and understand the theory is to try and implement ''the Math'' in practice :). Having used Scipy before, I thought there should be a similar package for Digital Communications in Python. This is a start!

What programming languages do you use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CommPy uses Python as its base programming language and python packages like NumPy, SciPy and Matplotlib. Some algorithms which are too slow in a pure Python implementation are implemented using Cython.

How can I contribute?
~~~~~~~~~~~~~~~~~~~~~
Implement any feature you want and send me a pull request :). If you want to suggest new features or discuss anything related to CommPy, please get in touch with me (veeresht@gmail.com).

How do I use CommPy?
~~~~~~~~~~~~~~~~~~~~
Requirements/Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^
- Python 2.7 or above
- NumPy 1.6 or above
- SciPy 0.10 or above
- Matplotlib 1.1 or above
- Cython 0.15 or above

Installation
^^^^^^^^^^^^

- Clone from github and install as follows (recommended)::

                $ git clone https://github.com/veeresht/CommPy.git
                $ cd CommPy
                $ sudo python setup.py build_ext --inplace
                $ sudo python setup.py install 

- To install using pip or easy_install use the following commands::
        
                $ sudo pip install scikit-commpy
                $ sudo easy_install scikit-commpy 


** I would greatly appreciate your feedback if you have found CommPy useful. Just send me a mail at veeresht.gmail.com **


For more details on CommPy, please visit http://veeresht.github.com/CommPy

.. _here: https://trello.com/board/commpy/4f44785f28107d10684bbd7d 
