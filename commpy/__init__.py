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
from commpy.filters import *
from commpy.modulation import *
from commpy.impairments import *
from commpy.sequences import *
from commpy.channels import *

try:
    from numpy.testing import Tester
    test = Tester().test
except:
    pass
