# Authors: CommPy contributors
# License: BSD 3-Clause

import math

import matplotlib.pyplot as plt
import numpy as np

import commpy.channels as chan
# ==================================================================================================
# Complete example using Commpy Wifi 802.11 physical parameters
# ==================================================================================================
from commpy.wifi80211 import Wifi80211

# AWGN channel
channels = chan.SISOFlatChannel(None, (1 + 0j, 0j))

w2 = Wifi80211(mcs=2)
w3 = Wifi80211(mcs=3)

# SNR range to test
SNRs2 = np.arange(0, 6) + 10 * math.log10(w2.get_modem().num_bits_symbol)
SNRs3 = np.arange(0, 6) + 10 * math.log10(w3.get_modem().num_bits_symbol)

BERs_mcs2 = w2.link_performance(channels, SNRs2, 10, 10, 600, stop_on_surpass_error=False)
BERs_mcs3 = w3.link_performance(channels, SNRs3, 10, 10, 600, stop_on_surpass_error=False)

# Test
plt.semilogy(SNRs2, BERs_mcs2, 'o-', SNRs3, BERs_mcs3, 'o-')
plt.grid()
plt.xlabel('Signal to Noise Ration (dB)')
plt.ylabel('Bit Error Rate')
plt.legend(('MCS 2', 'MCS 3'))
plt.show()
