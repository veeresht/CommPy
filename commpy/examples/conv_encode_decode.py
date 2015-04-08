
# Authors: Veeresh Taranalli <veeresht@gmail.com>
# License: BSD 3-Clause

import numpy as np
import commpy.channelcoding.convcode as cc
from commpy.utilities import *

# =============================================================================
# Example showing the encoding and decoding of convolutional codes
# =============================================================================

# G(D) corresponding to the convolutional encoder
generator_matrix = np.array([[05, 07]])
#generator_matrix = np.array([[03, 00, 02], [07, 04, 06]])

# Number of delay elements in the convolutional encoder
M = np.array([2])

# Create trellis data structure
trellis = cc.Trellis(M, generator_matrix)

# Traceback depth of the decoder
tb_depth = 5*(M.sum() + 1)

for i in range(10):
    # Generate random message bits to be encoded
    message_bits = np.random.randint(0, 2, 1000)

    # Encode message bits
    coded_bits = cc.conv_encode(message_bits, trellis)

    # Introduce bit errors (channel)
    #coded_bits[4] = 0
    #coded_bits[7] = 0

    # Decode the received bits
    decoded_bits = cc.viterbi_decode(coded_bits.astype(float), trellis, tb_depth)

    num_bit_errors = hamming_dist(message_bits, decoded_bits[:-M])
    #num_bit_errors = 1

    if num_bit_errors !=0:
        #print num_bit_errors, "Bit Errors found!"
        #print message_bits
        #print decoded_bits[tb_depth+3:]
        #print decoded_bits
        break
    else:
        print "No Bit Errors :)"

#print "==== Message Bits ==="
#print message_bits
#print "==== Coded Bits ====="
#print coded_bits
#print "==== Decoded Bits ==="
#print decoded_bits[tb_depth:]
