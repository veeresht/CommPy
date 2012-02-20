
#   Copyright 2012 Veeresh Taranalli <veeresht@gmail.com>
#
#   This file is part of CommPy.   
#
#   CommPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   CommPy is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""  
============================================
Pulse Shaping Filters (:mod:`commpy.filters`)
============================================

.. autosummary::
   :toctree: generated/
    
   rcosfilter          -- Class representing convolutional code trellis.
   rrcosfilter         -- Convolutional Encoder.
   gaussianfilter     -- Convolutional Decoder using the Viterbi algorithm.

"""

import numpy as np

__all__=['rcosfilter', 'rrcosfilter', 'gaussianfilter']

def rcosfilter(N, alpha, Ts, Fs):
    """
    Generates a raised cosine (RC) filter (FIR) impulse response.
    
    Parameters
    ----------
    N : int 
        Length of the filter in samples.

    alpha: float
        Roll off factor (Valid values are [0, 1]).

    Ts : float
        Symbol period in seconds.

    Fs : float 
        Sampling Rate in Hz.

    Returns
    -------

    h_rc : 1-D ndarray (float)
        Impulse response of the raised cosine filter.

    time_idx : 1-D ndarray (float) 
        Array containing the time indices, in seconds, for the impulse response.
    """

    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2))*T_delta
    sample_num = np.arange(N)
    h_rc = np.zeros(N, dtype=float)
        
    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rc[x] = 1.0
        elif alpha != 0 and t == Ts/(2*alpha):
            h_rc[x] = (np.pi/4)*(np.sin(np.pi*t/Ts)/(np.pi*t/Ts)) 
        elif alpha != 0 and t == -Ts/(2*alpha):
            h_rc[x] = (np.pi/4)*(np.sin(np.pi*t/Ts)/(np.pi*t/Ts))
        else:
            h_rc[x] = (np.sin(np.pi*t/Ts)/(np.pi*t/Ts))* \
                    (np.cos(np.pi*alpha*t/Ts)/(1-(((2*alpha*t)/Ts)*((2*alpha*t)/Ts))))
    
    return time_idx, h_rc  

def rrcosfilter(N, alpha, Ts, Fs):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.
    
    Parameters
    ----------
    N : int 
        Length of the filter in samples.
    
    alpha: float
        Roll off factor (Valid values are [0, 1]).
    
    Ts : float
        Symbol period in seconds.
    
    Fs : float 
        Sampling Rate in Hz.
    
    Returns
    ---------

    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.
    
    time_idx : 1-D ndarray of floats 
        Array containing the time indices, in seconds, for 
        the impulse response.
    """

    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2))*T_delta
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=float)
        
    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
        elif alpha != 0 and t == Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        elif alpha != 0 and t == -Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/Ts) +  \
                    4*alpha*(t/Ts)*np.cos(np.pi*t*(1+alpha)/Ts))/ \
                    (np.pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts)
        
    return time_idx, h_rrc

def gaussianfilter(N, alpha, Ts, Fs):
    """
    Generates a gaussian filter (FIR) impulse response.
    
    Parameters
    ----------

    N : int 
        Length of the filter in samples.

    alpha: float
        Roll off factor (Valid values are [0, 1]).
    
    Ts : float
        Symbol period in seconds.
    
    Fs : float 
        Sampling Rate in Hz.
    
    Returns
    -------

    h_gaussian : 1-D ndarray of floats
        Impulse response of the gaussian filter.
    
    time_index : 1-D ndarray of floats 
        Array containing the time indices for the impulse response.
    """
   
    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2))*T_delta
    h_gaussian = (np.sqrt(np.pi)/alpha)*np.exp(-((np.pi*time_index/alpha)*(np.pi*time_index/alpha)))
        
    return time_idx, h_gaussian  
