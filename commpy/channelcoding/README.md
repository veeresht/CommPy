# Channel codes basics

## Main idea

The main idea of the channel codes can be formulated as following thesises:
- **noise immunity** of the signal should be increased;
- **redundant bits** are added for *error detection* and *error correction*;
- some special algorithms (<u>coding schemes</u>) are used for this.

![](https://raw.githubusercontent.com/kirlf/CSP/master/FEC/assets/FECmainidea1.png)

The fact how "further" a certain algorithm divides the code words among themselves, and determines how strongly it protects the signal from noise [1, p.23].

![](https://habrastorage.org/webt/n7/o4/bs/n7o4bsf7_htlv10gsatc-yojbrq.png)

In the case of binary codes, the minimum distance between all existing code words is called **Hamming distance** and is usually denoted **dmin**:

<img src="https://raw.githubusercontent.com/kirlf/CSP/master/FEC/assets/FECexamp2.png" alt="examp2" width="400"/>


## Classification

Some classification is needed to talk about those or other implementations of the encoding and decoding algorithms.

First, the channel codes:
- can only [*detect*](https://en.wikipedia.org/wiki/Cyclic_redundancy_check) the presence of errors
- and they can also [*correct* errors](https://en.wikipedia.org/wiki/Error_correction_code).

Secondly, codes can be classified as **block** and **continuous**:

![](https://raw.githubusercontent.com/kirlf/CSP/master/FEC/assets/BlockCont.png)

## Net bit rate
Redundancy of the channel coding schemes influences (decreases) bit rate. Actually, it is the cost for the noiseless increasing.
[**Net bit rate**](https://en.wikipedia.org/wiki/Bit_rate#Information_rate) concept is usually used:

<img src="https://raw.githubusercontent.com/kirlf/CSP/master/FEC/assets/nebitrate.png" alt="net" width="500"/>

To change the code rate (k/n) of the block code dimensions of the Generator matrix can be changed:
![blockcoderate](https://raw.githubusercontent.com/kirlf/CSP/master/FEC/assets/coderateblock.png)

To change the coderate of the continuous code, e.g. convolutional code, **puncturing** procedure is frequently used:

![punct](https://raw.githubusercontent.com/kirlf/CSP/master/FEC/assets/punct.png)

## Example

Let us consider implematation of the **convolutional codes** as an example.

Main modeling routines:
- generate random message
- encode
- modulate it
- add a noise (e.g. AWGN)
- demodulate
- decode
- check the error correction

```python
import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.modulation as modulation

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
rate = 1/2 # code rate
L = 7 # constraint length
m = np.array([L-1]) # number of delay elements
generator_matrix = np.array([[0o171, 0o133]]) # generator branches
trellis = cc.Trellis(M, generator_matrix) # Trellis structure
```

Viterbi decoder parameters:

```python
tb_depth = 5*(m.sum() + 1) # traceback depth
```

Simulation loop:

```python
EbNo = 5 # energy per bit to noise power spectral density ratio (in dB)
snrdB = EbNo + 10*np.log10(k*rate) # Signal-to-Noise ratio (in dB)
noiseVar = 10**(-snrdB/10) # noise variance (power)


```

### Reference

[1] Moon, Todd K. "Error correction coding." Mathematical Methods and Algorithms. Jhon Wiley and Son (2005).
