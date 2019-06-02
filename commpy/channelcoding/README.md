# Channel codes basics

## Main idea

The main idea of the channel codes can be formulated as following thesises:
- **noise immunity** of the signal should be increased;
- **redundant bits** are added for *error detection* and *error correction*;
- some special algorithms (<u>coding schemes</u>) are used for this.

![](https://raw.githubusercontent.com/kirlf/CSP/master/FEC/assets/FECmainidea1.png)

The fact how "further" a certain algorithm divides the code words among themselves, and determines how strongly it protects the signal from noise [1, p.23].

![](https://habrastorage.org/webt/n7/o4/bs/n7o4bsf7_htlv10gsatc-yojbrq.png)

In the case of binary codes, the minimum distance between all existing code words is called ** Hamming distance ** and is usually denoted **dmin**:

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
**Net bit** rate concept is usually used:

<img src="https://raw.githubusercontent.com/kirlf/CSP/master/FEC/assets/nebitrate.png" alt="net" width="500"/>

To change the code rate (k/n) of the block code dimensions of the Generator matrix can be changed:
![blockcoderate](https://raw.githubusercontent.com/kirlf/CSP/master/FEC/assets/coderateblock.png)

To change the coderate of the continuous code, e.g. [convolutional code](https://github.com/kirlf/CSP/blob/master/FEC/Convolutional%20codes%20intro.md), **puncturing** procedure is frequently used:

![punct](https://raw.githubusercontent.com/kirlf/CSP/master/FEC/assets/punct.png)

### Reference

[1] Moon, Todd K. "Error correction coding." Mathematical Methods and Algorithms. Jhon Wiley and Son (2005).
