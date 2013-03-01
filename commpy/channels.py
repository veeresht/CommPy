
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

from numpy import complex, sum, pi, arange, array, size, shape, real, sqrt 
from numpy import matrix, sqrt, sum, zeros, concatenate, sinc
from numpy.random import randn, seed, random
from scipy.special import gamma, jn     
from scipy.signal import hamming, convolve, resample
from scipy.fftpack import ifft, fftshift, fftfreq
from scipy.interpolate import interp1d

def bec(input_bits, p_e):
    """ Binary Erasure Channel.

    Parameters
    __________
    input_bits : 1D ndarray containing {0, 1}
        Input arrary of bits to the channel.

    p_e : float in [0, 1]
        Erasure probability of the channel.

    Returns
    _______
    output_bits : 1D ndarray containing {0, 1}
        Output bits from the channel.
    """
    output_bits = input_bits.copy()
    output_bits[random(len(output_bits)) <= p_e] = -1
    return output_bits

def bsc(input_bits, p_t):
    """ Binary Symmetric Channel.

    Parameters
    __________
    input_bits : 1D ndarray containing {0, 1}
        Input arrary of bits to the channel.

    p_t : float in [0, 1]
        Transition/Error probability of the channel.

    Returns
    _______
    output_bits : 1D ndarray containing {0, 1}
        Output bits from the channel.
    """
    output_bits = input_bits.copy()
    flip_locs = (random(len(output_bits)) <= p_t)
    output_bits[flip_locs] = 1 ^ output_bits[flip_locs]
    return output_bits

def awgn(input_signal, snr_dB, rate=1.0):
    """ Addditive White Gaussian Noise (AWGN) Channel.

    Parameters
    __________
    input_signal : 1D ndarray of floats
        Input signal to the channel.

    snr_dB : float 
        Output SNR required in dB. 
    
    rate : float
        Rate of the a FEC code used if any, otherwise 1.

    Returns
    _______
    output_signal : 1D ndarray of floats 
        Output signal from the channel with the specified SNR.
    """

    avg_energy = sum(input_signal * input_signal)/len(input_signal)
    snr_linear = 10**(snr_dB/10.0)
    noise_variance = avg_energy/(2*rate*snr_linear)
    
    if input_signal.dtype is complex:
        noise = sqrt(noise_variance) * randn(len(input_signal)) * (1+1j)
    else:
        noise = sqrt(2*noise_variance) * randn(len(input_signal))
    
    output_signal = input_signal + noise
    
    return output_signal 







# =============================================================================
# Incomplete code to implement fading channels
# =============================================================================

#def doppler_jakes(max_doppler, filter_length):

#    fs = 32.0*max_doppler
#    ts = 1/fs
#    m = arange(0, filter_length/2)
    
    # Generate the Jakes Doppler Spectrum impulse response h[m]    
#    h_jakes_left = (gamma(3.0/4) * 
#                    pow((max_doppler/(pi*abs((m-(filter_length/2))*ts))), 0.25) *
#                    jn(0.25, 2*pi*max_doppler*abs((m-(filter_length/2))*ts)))
#    h_jakes_center = array([(gamma(3.0/4)/gamma(5.0/4)) * pow(max_doppler, 0.5)])
#    h_jakes = concatenate((h_jakes_left[0:filter_length/2-1], 
#                     h_jakes_center, h_jakes_left[::-1]))
#    h_jakes = h_jakes*hamming(filter_length)
#    h_jakes = h_jakes/(sum(h_jakes**2)**0.5) 

# -----------------------------------------------------------------------------
#    jakes_psd_right = (1/(pi*max_doppler*(1-(freqs/max_doppler)**2)**0.5))**0.5
#    zero_pad = zeros([(fft_size-filter_length)/2, ]) 
#    jakes_psd = concatenate((zero_pad, jakes_psd_right[::-1],  
#                             jakes_psd_right, zero_pad))
    #print size(jakes_psd)
#    jakes_impulse = real(fftshift(ifft(jakes_psd, fft_size)))
#    h_jakes = jakes_impulse[(fft_size-filter_length)/2 + 1 : (fft_size-filter_length)/2 + filter_length + 1]
#    h_jakes = h_jakes*hamming(filter_length)
#    h_jakes = h_jakes/(sum(h_jakes**2)**0.5)
# -----------------------------------------------------------------------------
#   return h_jakes

#def rayleigh_channel(ts_input, max_doppler, block_length, path_gains, 
#                     path_delays):
    
#    fs_input = 1.0/ts_input
#    fs_channel = 32.0*max_doppler
#    ts_channel = 1.0/fs_channel
#    interp_factor = fs_input/fs_channel
#    channel_length = block_length/interp_factor
#    n1 = -10    
#    n2 = 10    

#   filter_length = 1024

    # Generate the Jakes Doppler Spectrum impulse response h[m]    
#    h_jakes = doppler_jakes(max_doppler, filter_length)
    
    # Generate the complex Gaussian Random Process
#    g_var = 0.5
#    gain_process = zeros([len(path_gains), block_length], dtype=complex)    
#    delay_process = zeros([n2+1-n1, len(path_delays)])    
#    for k in xrange(len(path_gains)):
#        g = (g_var**0.5) * (randn(channel_length) + 1j*randn(channel_length))
#        g_filt = convolve(g, h_jakes, mode='same')
#        g_filt_interp = resample(g_filt, block_length)
#        gain_process[k,:] = pow(10, (path_gains[k]/10.0)) * g_filt_interp
#        delay_process[:,k] = sinc((path_delays[k]/ts_input) - arange(n1, n2+1))
     
    #channel_matrix = 0           
#    channel_matrix = matrix(delay_process)*matrix(gain_process)
        
#    return channel_matrix, gain_process, h_jakes
    

    
    
        
    
    
