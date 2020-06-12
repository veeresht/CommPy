# first code : 08-06-20
# firefly detector

from numpy.linalg import qr
import numpy as np
import time
from commpy import MIMOFlatChannel, QAMModem, kbest
from commpy.links import link_performance, LinkModel
import matplotlib.pyplot as plt


def firefly(y, h, nb_iter = 100, gamma = 0.5, k = 1):

    # number of transmit antennas & receive antennas
    nb_tx, nb_rx = h.shape
    N = nb_tx

    # construction of optimal vector and constructed vector
    x_opt = np.empty(nb_rx)
    x = np.empty(nb_rx)

    # construction Euclidien distance list for the 2 cases
    ud = np.zeros(2)

    # construction attractiveness list
    beta = np.zeros(2)

    # QR decomposition
    q, r = qr(h)
    yt = q.T.dot(y)

    # Computing !
    # print("Computing !")
    # usually number of transmit antennas is not different than those receive
    E_opt = np.inf  # E(x_opt)
    iter = 0

    while iter < nb_iter :

        for i in range(N-1,-1,-1):

            # compute the Euclidien distance (equ 16)
            sum_temp = 0
            for j in range(i+1,N):
                sum_temp = sum_temp + r[i][j] * x[j]

            xi = -1  # for first case
            ud[0] = (yt[i] - r[i][i] * xi - sum_temp) ** 2  # todo
            xi =  1  # for second case
            ud[1] = (yt[i] - r[i][i] * xi - sum_temp) ** 2  # todo

            # Compute attractiveness parameter (equ 17)
            beta[0] = np.exp(-gamma * ud[0] ** k) # todo
            beta[1] = np.exp(-gamma * ud[1] ** k) # todo

            # Compute probability metric (equ 18)
            p = beta[0] / ( beta[0] + beta[1] )  # todo

            # generate uniformly random variable called alpha
            alpha = np.random.random()

            # calculate xi value (equ 19)
            if p > alpha :
                x[i] = -1
            else:
                x[i] = 1

        # update x_opt (equ 22)
        E_temp = 0.   # E(x)

        for i in range(N-1,-1,-1):
            sum_temp = 0
            for j in range(i+1,N):
                sum_temp = sum_temp + r[i][j] * x[j]
            E_temp = E_temp + (yt[i] - r[i][i] * x[i] - sum_temp) ** 2

        if E_opt > E_temp:
            x_opt = x
            E_opt = E_temp


        iter = iter + 1     # increment the iterator
        #print("We are in ", iter," iteration and remain ", nb_iter - iter)

    return x_opt




################################################################################
# testing the code

# Same SNRs for every model
SNRs = np.arange(0, 21, 5) + 10 * np.log10(4)

############################################
# Model
############################################
modem = QAMModem(2)
modem.constellation = -1, 1
receivers_str = ('KSE-16', 'firefly')
channels = tuple(MIMOFlatChannel(8, 8) for i in range(2))

############################################
# Set channel fading
############################################
for i in range(2):
    channels[i].uncorr_rayleigh_fading(float)   # or complex for a complex canal

############################################
# Functions
############################################
def KSE16(y, h, constellation, t):
    return modem.demodulate(kbest(y, h, constellation, 16), 'hard')

def FA(y, h, constellation, t):
    return modem.demodulate(firefly(y, h), 'hard')


modulates = tuple(modem.modulate for _ in range(2))
modems = (modem,) * 2

receivers = (KSE16, FA)

############################################
# Link_performance
############################################
nb_it = 500*103
nb_err = 500
chunk = 5000


############################################
# Models
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


# Compute & plot results of two detectors
#print("Computing KSE-16")
#BERs0 = perf(models[0])
print("Computing FA")
start = time.clock()
BERs1 = perf(models[1])
end = time.clock()
print("Finish computing in ", (end - start)/60,"min")

#plt.semilogy(10*np.log10(SNRs), BERs0, label = "KSE-16")
plt.semilogy(10*np.log10(SNRs), BERs1, label = "FA")

plt.xlabel("SNRs (dB)")
plt.ylabel("BER")
plt.legend()
plt.show()