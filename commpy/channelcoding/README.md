# Channel codes basics

## Main idea

The main idea of the channel codes can be formulated as following thesises:
- **noise immunity** of the signal should be increased;
- **redundant bits** are added for *error detection* and *error correction*;
- some special algorithms (<u>coding schemes</u>) are used for this.

<img src="https://raw.githubusercontent.com/veeresht/CommPy/master/commpy/channelcoding/doc/assets/FECmainidea1.png" width="800" />

The fact how "further" a certain algorithm divides the code words among themselves, and determines how strongly it protects the signal from noise [1, p.23].

<img src="https://habrastorage.org/webt/n7/o4/bs/n7o4bsf7_htlv10gsatc-yojbrq.png" width="800" />

In the case of binary codes, the minimum distance between all existing code words is called **Hamming distance** and is usually denoted **dmin**:

<img src="https://raw.githubusercontent.com/veeresht/CommPy/master/commpy/channelcoding/doc/assets/FECexamp2.png" alt="examp2" width="400"/>


## Classification

Some classification is needed to talk about those or other implementations of the encoding and decoding algorithms.

First, the channel codes:
- can only [*detect*](https://en.wikipedia.org/wiki/Cyclic_redundancy_check) the presence of errors
- and they can also [*correct* errors](https://en.wikipedia.org/wiki/Error_correction_code).

Secondly, codes can be classified as **block** and **continuous**:

![](https://raw.githubusercontent.com/veeresht/CommPy/master/commpy/channelcoding/doc/assets/BlockCont.png)

## Net bit rate
Redundancy of the channel coding schemes influences (decreases) bit rate. Actually, it is the cost for the noiseless increasing.
[**Net bit rate**](https://en.wikipedia.org/wiki/Bit_rate#Information_rate) concept is usually used:

<img src="https://raw.githubusercontent.com/veeresht/CommPy/master/commpy/channelcoding/doc/assets/nebitrate.png" alt="net" width="500"/>

To change the code rate (k/n) of the block code dimensions of the Generator matrix can be changed:
![blockcoderate](https://raw.githubusercontent.com/veeresht/CommPy/master/commpy/channelcoding/doc/assets/coderateblock.png)

To change the coderate of the continuous code, e.g. convolutional code, **puncturing** procedure is frequently used:

![punct](https://raw.githubusercontent.com/veeresht/CommPy/master/commpy/channelcoding/doc/assets/punct.png)

## Example

Let us consider implematation of the **convolutional codes** as an example:

<img src="https://habrastorage.org/webt/v3/v5/w2/v3v5w2gbwk34nzk_2qt25baoebq.png" width="500"/>

*Main modeling routines: random message genaration, channel encoding, baseband modulation, additive noise (e.g. AWGN), baseband demodulation, channel decoding, BER calculation.*

```python
import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.modulation as modulation

def BER_calc(a, b):
    num_ber = np.sum(np.abs(a - b))
    ber = np.mean(np.abs(a - b))
    return int(num_ber), ber

N = 100000 #number of symbols per the frame
message_bits = np.random.randint(0, 2, N) # message

M = 4 # modulation order (QPSK)
k = np.log2(M) #number of bit per modulation symbol
modem = modulation.PSKModem(M) # M-PSK modem initialization 
```

The [following](https://en.wikipedia.org/wiki/File:Conv_code_177_133.png) convolutional code will be used:

![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Conv_code_177_133.png/800px-Conv_code_177_133.png)

*Shift-register for the (7, [171, 133]) convolutional code polynomial.*

Convolutional encoder parameters:

```python
generator_matrix = np.array([[5, 7]]) # generator branches
trellis = cc.Trellis(np.array([M]), generator_matrix) # Trellis structure

rate = 1/2 # code rate
L = 7 # constraint length
m = np.array([L-1]) # number of delay elements
```

Viterbi decoder parameters:

```python
tb_depth = 5*(m.sum() + 1) # traceback depth
```

Two oppitions of the Viterbi decoder will be tested:
- *hard* (hard inputs)
- *unquatized* (soft inputs)

Additionally, uncoded case will be considered. 

Simulation loop:

```python
EbNo = 5 # energy per bit to noise power spectral density ratio (in dB)
snrdB = EbNo + 10*np.log10(k*rate) # Signal-to-Noise ratio (in dB)
noiseVar = 10**(-snrdB/10) # noise variance (power)

N_c = 10 # number of trials

BER_soft = np.zeros(N_c)
BER_hard = np.zeros(N_c)
BER_uncoded = np.zeros(N_c)

for cntr in range(N_c):
    
    message_bits = np.random.randint(0, 2, N) # message
    coded_bits = cc.conv_encode(message_bits, trellis) # encoding
    
    modulated = modem.modulate(coded_bits) # modulation
    modulated_uncoded = modem.modulate(message_bits) # modulation (uncoded case)

    Es = np.mean(np.abs(modulated)**2) # symbol energy
    No = Es/((10**(EbNo/10))*np.log2(M)) # noise spectrum density

    noisy = modulated + np.sqrt(No/2)*\
        (np.random.randn(modulated.shape[0])+\
         1j*np.random.randn(modulated.shape[0])) # AWGN
    
    noisy_uncoded = modulated_uncoded + np.sqrt(No/2)*\
        (np.random.randn(modulated_uncoded.shape[0])+\
         1j*np.random.randn(modulated_uncoded.shape[0])) # AWGN (uncoded case)

    demodulated_soft = modem.demodulate(noisy, demod_type='soft', noise_var=noiseVar) # demodulation (soft output)
    demodulated_hard = modem.demodulate(noisy, demod_type='hard') # demodulation (hard output)
    demodulated_uncoded = modem.demodulate(noisy_uncoded, demod_type='hard') # demodulation (uncoded case)

    decoded_soft = cc.viterbi_decode(demodulated_soft, trellis, tb_depth, decoding_type='unquantized') # decoding (soft decision)
    decoded_hard = cc.viterbi_decode(demodulated_hard, trellis, tb_depth, decoding_type='hard') # decoding (hard decision)

    NumErr, BER_soft[cntr] = BER_calc(message_bits, decoded_soft[:message_bits.size]) # bit-error ratio (soft decision)
    NumErr, BER_hard[cntr] = BER_calc(message_bits, decoded_hard[:message_bits.size]) # bit-error ratio (hard decision)
    NumErr, BER_uncoded[cntr] = BER_calc(message_bits, demodulated_uncoded[:message_bits.size]) # bit-error ratio (uncoded case)

mean_BER_soft = BER_soft.mean() # averaged bit-error ratio (soft decision)
mean_BER_hard = BER_hard.mean() # averaged bit-error ratio (hard decision)
mean_BER_uncoded = BER_uncoded.mean() # averaged bit-error ratio (uncoded case)

print("Soft decision:\n{}\n".format(mean_BER_soft))
print("Hard decision:\n{}\n".format(mean_BER_hard))
print("Uncoded message:\n{}\n".format(mean_BER_uncoded))
```

Outputs:

```python
Soft decision:
2.8000000000000003e-05

Hard decision:
0.0007809999999999999

Uncoded message:
0.009064
```

### Reference

[1] Moon, Todd K. "Error correction coding." Mathematical Methods and Algorithms. Jhon Wiley and Son (2005).
