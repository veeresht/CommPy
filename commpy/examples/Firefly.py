# first code : 12-06-20
# firefly detector

from numpy.linalg import qr
import numpy as np
import time, math
from commpy import MIMOFlatChannel, QAMModem, kbest
from commpy.links import link_performance, LinkModel
import matplotlib.pyplot as plt


def firefly(y, h, nb_iter = 100, gamma = 0.5, k = 1):

    if isinstance(h[0, 0], complex):
        h = np.block([[h.real, -h.imag], [h.imag, h.real]])
        y = np.concatenate((y.real, y.imag))
        is_complex = True
    else:
        is_complex = False

    # number of transmit antennas & receive antennas
    nb_tx, nb_rx = h.shape
    N = nb_tx

    # construction of optimal vector and constructed vector
    x_opt = np.empty(N, dtype= np.int8)
    x = np.empty(N, dtype= np.int8)

    # construction Euclidien distance list for the 2 cases
    ud = np.zeros(2)

    # construction attractiveness list
    beta = np.zeros(2)

    # QR decomposition
    q, r = qr(h)
    yt = q.conj().T.dot(y)

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
            ud[0] = (yt[i] - r[i][i] * xi - sum_temp) ** 2
            xi =  1  # for second case
            ud[1] = (yt[i] - r[i][i] * xi - sum_temp) ** 2

            # Compute attractiveness parameter (equ 17)
            beta[0] = np.exp(-gamma * ud[0] ** k)
            beta[1] = np.exp(-gamma * ud[1] ** k)

            # Compute probability metric (equ 18)
            p = beta[0] / ( beta[0] + beta[1] )

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
            x_opt = x.copy()
            E_opt = E_temp


        iter = iter + 1     # increment the iterator
        #print("We are in ", iter," iteration and remain ", nb_iter - iter)
    if is_complex:
        return x_opt[:int(N/2)]+1j*x_opt[int(N/2):]
    else:
        return x_opt




################################################################################
# testing the code


############################################
# Model
############################################
modem = QAMModem(4)
#modem.constellation = -1, 1
receivers_str = ('KSE-16', 'firefly')
channels = tuple(MIMOFlatChannel(8, 8) for i in range(5))



# Same SNRs for every model
SNRs = np.arange(0, 14, 2) #+ 10 * np.log10(modem.num_bits_symbol)

############################################
# Set channel fading
############################################
for i in range(5):
    channels[i].uncorr_rayleigh_fading(complex)   # or complex for a complex canal

############################################
# Functions
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

modulates = tuple(modem.modulate for _ in range(5))
modems = (modem,) * 5

receivers = (KSE16, FA20, FA40, FA60, FA100)

############################################
# Link_performance
############################################
nb_err = 200
nb_it = math.ceil(nb_err / 4e-4)
chunk = 1440


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

print("Computing KSE-16")
start = time.clock()
BERs0 = perf(models[0])
end1 = time.clock()
print("Finish computing in ", (end1 - start)/60,"min")

print("Computing FA20")
BERs1 = perf(models[1])
end2 = time.clock()
print("Finish computing in ", (end2 - end1)/60,"min")

print("Computing FA40")
BERs2 = perf(models[2])
end3 = time.clock()
print("Finish computing in ", (end3 - end2)/60,"min")

print("Computing FA60")
BERs3 = perf(models[3])
end4 = time.clock()
print("Finish computing in ", (end4 - end3)/60,"min")

print("Computing FA100")
BERs4 = perf(models[4])
end5 = time.clock()
print("Finish computing in ", (end5 - end4)/60,"min")


#plotting

plt.semilogy(10*np.log10(SNRs), BERs0,'-*', label = "KSE-16")
plt.semilogy(10*np.log10(SNRs), BERs1,'-*' ,label = "FA, iT = 20")
plt.semilogy(10*np.log10(SNRs), BERs2,'-*' ,label = "FA, iT = 40")
plt.semilogy(10*np.log10(SNRs), BERs3,'-*' ,label = "FA, iT = 60")
plt.semilogy(10*np.log10(SNRs), BERs4,'-*' ,label = "FA, iT = 100")

plt.xlabel("SNRs (dB)")
plt.ylabel("BER")
plt.legend()
plt.grid()
plt.show()

temps = [end2 - end1, end3 - end2, end4 - end3, end5 - end4]
it = [20, 40, 60, 100]
plt.plot(it, temps,"-*" ,label = "temporal complexity")
plt.xlabel("number of iterations")
plt.ylabel("time in sec")
plt.legend()
plt.grid()
plt.show()