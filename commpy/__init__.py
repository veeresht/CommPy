"""
CommPy
================================================


Contents
--------

Subpackages
-----------
::

 channelcoding                --- Channel Coding Algorithms [*]

"""
#from channelcoding import *
from .filters import *
from .modulation import *
from .impairments import *
from .sequences import *
from .channels import *

try:
    from numpy.testing import Tester
    test = Tester().test
except:
    pass
