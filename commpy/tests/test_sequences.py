# Authors: CommPy contributors
# License: BSD 3-Clause

import numpy as np
from numpy import array
from numpy.testing import run_module_suite, assert_raises, assert_equal, assert_almost_equal

from commpy.sequences import pnsequence, zcsequence

def test_pnsequence():
    # Test the raises of errors
    with assert_raises(ValueError):
        pnsequence(4, '001', '1101', 2**4 - 1)
    with assert_raises(ValueError):
        pnsequence(4, '0011', '110', 2 ** 4 - 1)

    # Test output with
    assert_equal(pnsequence(4, '0011', [1, 1, 0, 1], 7), array((1, 1, 0, 0, 1, 0, 1), int),
                 err_msg='Pseudo-noise sequence is not the one expected.')
    assert_equal(pnsequence(4, (0, 0, 1, 1), array((1, 1, 0, 1)), 7), array((1, 1, 0, 0, 1, 0, 1), int),
                 err_msg='Pseudo-noise sequence is not the one expected.')

def test_zcsequence():
    # Test the raises of errors
    with assert_raises(ValueError):
        zcsequence(u=-1, seq_length=20, q=0)
    with assert_raises(ValueError):
        zcsequence(u=20, seq_length=0, q=0)
    with assert_raises(ValueError):
        zcsequence(u=3, seq_length=18, q=0)
    with assert_raises(ValueError):
        zcsequence(u=3.1, seq_length=11, q=0)
    with assert_raises(ValueError):
        zcsequence(u=3, seq_length=11.1, q=0)
    with assert_raises(ValueError):
        zcsequence(u=3, seq_length=11, q=0.1)

    # Test output with
    assert_almost_equal(zcsequence(u=1, seq_length=2, q=0), array([1.000000e+00+0.j, 6.123234e-17-1.j]),
        err_msg='CAZAC sequence is not the expected one.')

    # Test if output cross-correlation is valid
    seqCAZAC = zcsequence(u=3, seq_length=20, q=0)
    x = np.fft.fft(seqCAZAC) / np.sqrt(seqCAZAC.size)
    h = (np.fft.ifft(np.conj(x) * x)*np.sqrt(seqCAZAC.size)).T
    corr = np.absolute(h)**2/h.size
    assert_almost_equal(corr[0], 1.,
        err_msg='CAZAC sequence auto-correlation is not valid, first term is not 1')
    assert_almost_equal(corr[1:], np.zeros(corr.size-1),
        err_msg='CAZAC sequence auto-correlation is not valid, all terms except first are not 0')

if __name__ == "__main__":
    run_module_suite()
