# Authors: CommPy contributors
# License: BSD 3-Clause

import math

import matplotlib.pyplot as plt
import numpy as np

from commpy.channels import MIMOFlatChannel
from commpy.links import link_performance, LinkModel
from commpy.modulation import QAMModem, kbest, firefly

############################################
# Defining the model
############################################

# Modem: QPSK
modem = QAMModem(4)
modulates = (modem.modulate,) * 5
modems = (modem,) * 5

# Channels: 8x8 MIMO
channels = tuple(MIMOFlatChannel(8, 8) for _ in range(5))

# Set channel fading
for i in range(5):
    channels[i].uncorr_rayleigh_fading(complex)

# Same SNRs for every model
SNRs = np.arange(0, 14, 2) + 10 * np.log10(modem.num_bits_symbol)


############################################
# Define receivers
############################################
def KSE16(y, h, constellation, t):
    return modem.demodulate(kbest(y, h, constellation, 16), 'hard')


def FA20(y, h, constellation, t):
    return modem.demodulate(firefly(y, h, 20), 'hard')


def FA40(y, h, constellation, t):
    return modem.demodulate(firefly(y, h, 40), 'hard')


def FA60(y, h, constellation, t):
    return modem.demodulate(firefly(y, h, 60), 'hard')


def FA100(y, h, constellation, t):
    return modem.demodulate(firefly(y, h, 100), 'hard')


receivers_str = ('KSE-16', 'FA-20', 'FA-40', 'FA-60', 'FA-100')
receivers = (KSE16, FA20, FA40, FA60, FA100)

############################################
# Define simulation parameters
############################################
nb_err = 200
nb_it = math.ceil(nb_err / 4e-4)
chunk = 1440

############################################
# Build model
############################################
models = []
for i in range(len(modems)):
    models.append(LinkModel(modulates[i], channels[i], receivers[i],
                            modems[i].num_bits_symbol, modems[i].constellation, modems[i].Es))


############################################
# Test
############################################
def perf(model):
    return link_performance(model, SNRs, nb_it, nb_err, chunk)


############################################
# Compute
############################################
print("Computing KSE-16")
BERs0 = perf(models[0])
print("Finish computing")

print("Computing FA20")
BERs1 = perf(models[1])
print("Finish computing")

print("Computing FA40")
BERs2 = perf(models[2])
print("Finish computing")

print("Computing FA60")
BERs3 = perf(models[3])
print("Finish computing")

print("Computing FA100")
BERs4 = perf(models[4])
print("Finish computing")

############################################
# Plotting
############################################
plt.figure()
plt.semilogy(SNRs, BERs0, '-*', label="KSE-16")
plt.semilogy(SNRs, BERs1, '-*', label="FA, iT = 20")
plt.semilogy(SNRs, BERs2, '-*', label="FA, iT = 40")
plt.semilogy(SNRs, BERs3, '-*', label="FA, iT = 60")
plt.semilogy(SNRs, BERs4, '-*', label="FA, iT = 100")

plt.xlabel("SNRs (dB)")
plt.ylabel("BER")
plt.legend()
plt.grid()
plt.show()
