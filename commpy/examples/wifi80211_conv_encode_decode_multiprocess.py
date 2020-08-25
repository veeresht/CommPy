# Authors: CommPy contributors
# License: BSD 3-Clause

import math
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np

import commpy.channels as chan
# ==================================================================================================
# Complete example using Commpy Wifi 802.11 physical parameters
# ==================================================================================================
from commpy.multiprocess_links import Wifi80211

# AWGN channel
channel = chan.SISOFlatChannel(None, (1 + 0j, 0j))

w2 = Wifi80211(mcs=2)
w3 = Wifi80211(mcs=3)

# SNR range to test
SNRs2 = np.arange(0, 6) + 10 * math.log10(w2.get_modem().num_bits_symbol)
SNRs3 = np.arange(0, 6) + 10 * math.log10(w3.get_modem().num_bits_symbol)


start = time.time()
BERs = w2.link_performance_mp_mcs([2, 3], [SNRs2, SNRs3], channel, 10, 10, 600, stop_on_surpass_error=False)
print(BERs)
print(str(timedelta(seconds=(time.time() - start))))
# Test
plt.semilogy(SNRs2, BERs[2][0], 'o-', SNRs3, BERs[3][0], 'o-')
plt.grid()
plt.xlabel('Signal to Noise Ration (dB)')
plt.ylabel('Bit Error Rate')
plt.legend(('MCS 2', 'MCS 3'))
plt.show()
