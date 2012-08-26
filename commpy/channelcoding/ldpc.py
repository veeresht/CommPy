
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

""" LDPC Codes """
from numpy import zeros, shape, tanh, arctanh, array, delete, prod, dot

__all__ = ['ldpc_decode']

def ldpc_decode(pcheck_matrix, rx_codeword, n_iters):
    """ 
    LDPC Decoder using belief propagation for an AWGN channel.

    Parameters
    ----------
    pcheck_matrix : 2D array of 0's and 1's
        Parity Check Matrix of the linear (LDPC) code.

    rx_codeword : 1D array of float
        Received codeword from an AWGN channel.

    n_iter : int
        Max. number of iterations of decoding to be done.

    Returns
    -------
    decoded_bits : 1D array of 0's and 1's
        The codeword after decoding.

    """
    [n_c_nodes, n_v_nodes] = shape(pcheck_matrix)
    comp_mat = zeros([n_c_nodes, n_v_nodes])
    llr_vals = rx_codeword
    for i in xrange(n_iters):
        # Check Node Update
        for v_node in xrange(n_v_nodes):
            for c_node in xrange(n_c_nodes):
                if pcheck_matrix[c_node, v_node] == 1:
                    prev_llrs = delete(llr_vals, v_node)
                    prev_llrs = prev_llrs[delete(pcheck_matrix[c_node, :], v_node) == 1]
                    prev_compvals = delete(comp_mat[c_node, :], v_node)
                    prev_compvals = prev_compvals[delete(pcheck_matrix[c_node, :], v_node) == 1]        
                    llr_prod = prod(tanh(-(prev_llrs - prev_compvals)/2))
                    comp_mat[c_node, v_node] = -2 * arctanh(llr_prod)
        
        # Variable Node Update
        for v_node in xrange(n_v_nodes):
            llr_vals[v_node] = llr_vals[v_node] + sum(comp_mat[:,v_node][pcheck_matrix[:,v_node]==1])

        decoded_bits = array(llr_vals > 0, dtype=int)
        if not (dot(pcheck_matrix, decoded_bits)%2).any():
            print "Perfect Decoding, # Iterations: " + str(i+1)
            break
    
    return decoded_bits

if __name__ == '__main__':

    pcheck_matrix = array([[1, 1, 1, 0, 0, 1, 1, 0, 0, 1], 
                           [1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
                           [0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
                           [0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                           [1, 1, 0, 1, 0, 0, 1, 1, 1, 0]])
    rx_codeword = array([-1.3, -1.7, -1.5, -0.08, 0.2, 1.9, -1.5, 1.3, -1.1, 1.2])
    n_iters = 10
    decoded_bits = ldpc_decode(pcheck_matrix, rx_codeword, n_iters)
        
    print decoded_bits


