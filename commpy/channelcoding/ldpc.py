# Authors: CommPy contributors
# License: BSD 3-Clause

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splg

__all__ = ['build_matrix', 'get_ldpc_code_params', 'ldpc_bp_decode', 'write_ldpc_params',
           'triang_ldpc_systematic_encode']

_llr_max = 500

def build_matrix(ldpc_code_params):
    """
    Build the parity check and generator matrices from parameters dictionary and add the result in this dictionary.
    Generator matrix is valid only for triangular systematic LDPC codes.

    Parameters
    ----------
    ldpc_code_params: dictionary that at least contains these parameters
        Parameters of the LDPC code:
            n_vnodes (int) - number of variable nodes.
            n_cnodes (int) - number of check nodes.
            max_cnode_deg (int) - maximal degree of a check node.
            cnode_adj_list (1D-ndarray of ints) - flatten array so that
                cnode_adj_list.reshape((n_cnodes, max_cnode_deg)) gives for each check node the adjacent variable nodes.
            cnode_deg_list (1D-ndarray of ints) - degree of each check node.

    Add
    ---
    to ldpc_code_param:
            parity_check_matrix (CSC sparse matrix of int8) - parity check matrix.
            generator_matrix (CSR sparse matrix) - generator matrix of the code.
    """
    n_cnodes = ldpc_code_params['n_cnodes']
    cnode_deg_list = ldpc_code_params['cnode_deg_list']
    cnode_adj_list = ldpc_code_params['cnode_adj_list'].reshape((n_cnodes, ldpc_code_params['max_cnode_deg']))

    parity_check_matrix = sp.lil_matrix((n_cnodes, ldpc_code_params['n_vnodes']), dtype=np.int8)
    for cnode_idx in range(n_cnodes):
        parity_check_matrix[cnode_idx, cnode_adj_list[cnode_idx, :cnode_deg_list[cnode_idx]]] = 1

    parity_check_matrix = parity_check_matrix.tocsc()
    systematic_part = parity_check_matrix[:, -n_cnodes:]
    parity_part = parity_check_matrix[:, :-n_cnodes]

    ldpc_code_params['parity_check_matrix'] = parity_check_matrix
    ldpc_code_params['generator_matrix'] = splg.inv(systematic_part).dot(parity_part).tocsr()


def get_ldpc_code_params(ldpc_design_filename, compute_matrix=False):
    """
    Extract parameters from LDPC code design file and produce an parity check matrix if asked.

    The file is structured as followed (examples are available in designs/ldpc/):
        n_vnode n_cnode
        max_vnode_deg max_cnode_deg
        List of the degree of each vnode
        List of the degree of each cnode
        For each vnode (line by line, separated by '\t'): index of the connected cnodes
        For each cnode (line by line, separated by '\t'): index of the connected vnodes

    Parameters
    ----------
    ldpc_design_filename : string
        Filename of the LDPC code design file.

    compute_matrix : boolean
        Specify if the parity check matrix must be computed.
        *Default* is False.

    Returns
    -------
    ldpc_code_params : dictionary that at least contains these parameters
        Parameters of the LDPC code:
            n_vnodes (int) - number of variable nodes.
            n_cnodes (int) - number of check nodes.
            max_vnode_deg (int) - maximal degree of a variable node.
            max_cnode_deg (int) - maximal degree of a check node.
            vnode_adj_list (1D-ndarray of ints) - flatten array so that
                vnode_adj_list.reshape((n_vnodes, max_vnode_deg)) gives for each variable node the adjacent check nodes.
            cnode_adj_list (1D-ndarray of ints) - flatten array so that
                cnode_adj_list.reshape((n_cnodes, max_cnode_deg)) gives for each check node the adjacent variable nodes.
            vnode_cnode_map (1D-ndarray of ints) - flatten array providing the mapping between vnode and cnode indexes.
            cnode_vnode_map (1D-ndarray of ints) - flatten array providing the mapping between vnode and cnode indexes.
            vnode_deg_list (1D-ndarray of ints) - degree of each variable node.
            cnode_deg_list (1D-ndarray of ints) - degree of each check node.
            parity_check_matrix (CSC sparse matrix of int8) - parity check matrix if asked.
    """

    with open(ldpc_design_filename) as ldpc_design_file:

        [n_vnodes, n_cnodes] = [int(x) for x in ldpc_design_file.readline().split(' ')]
        [max_vnode_deg, max_cnode_deg] = [int(x) for x in ldpc_design_file.readline().split(' ')]
        vnode_deg_list = np.array([int(x) for x in ldpc_design_file.readline().split(' ')[:-1]], np.int32)
        cnode_deg_list = np.array([int(x) for x in ldpc_design_file.readline().split(' ')[:-1]], np.int32)

        cnode_adj_list = -np.ones([n_cnodes, max_cnode_deg], int)
        vnode_adj_list = -np.ones([n_vnodes, max_vnode_deg], int)

        for vnode_idx in range(n_vnodes):
            vnode_adj_list[vnode_idx, 0:vnode_deg_list[vnode_idx]] = \
                np.array([int(x)-1 for x in ldpc_design_file.readline().split('\t')])

        for cnode_idx in range(n_cnodes):
            cnode_adj_list[cnode_idx, 0:cnode_deg_list[cnode_idx]] = \
                np.array([int(x)-1 for x in ldpc_design_file.readline().split('\t')])

    cnode_vnode_map = -np.ones([n_cnodes, max_cnode_deg], int)
    vnode_cnode_map = -np.ones([n_vnodes, max_vnode_deg], int)

    for cnode in range(n_cnodes):
        for i, vnode in enumerate(cnode_adj_list[cnode, 0:cnode_deg_list[cnode]]):
            cnode_vnode_map[cnode, i] = np.where(vnode_adj_list[vnode, :] == cnode)[0]

    for vnode in range(n_vnodes):
        for i, cnode in enumerate(vnode_adj_list[vnode, 0:vnode_deg_list[vnode]]):
            vnode_cnode_map[vnode, i] = np.where(cnode_adj_list[cnode, :] == vnode)[0]

    cnode_adj_list_1d = cnode_adj_list.flatten().astype(np.int32)
    vnode_adj_list_1d = vnode_adj_list.flatten().astype(np.int32)
    cnode_vnode_map_1d = cnode_vnode_map.flatten().astype(np.int32)
    vnode_cnode_map_1d = vnode_cnode_map.flatten().astype(np.int32)

    ldpc_code_params = {}

    ldpc_code_params['n_vnodes'] = n_vnodes
    ldpc_code_params['n_cnodes'] = n_cnodes
    ldpc_code_params['max_cnode_deg'] = max_cnode_deg
    ldpc_code_params['max_vnode_deg'] = max_vnode_deg
    ldpc_code_params['cnode_adj_list'] = cnode_adj_list_1d
    ldpc_code_params['cnode_vnode_map'] = cnode_vnode_map_1d
    ldpc_code_params['vnode_adj_list'] = vnode_adj_list_1d
    ldpc_code_params['vnode_cnode_map'] = vnode_cnode_map_1d
    ldpc_code_params['cnode_deg_list'] = cnode_deg_list
    ldpc_code_params['vnode_deg_list'] = vnode_deg_list

    if compute_matrix:
        build_matrix(ldpc_code_params)

    return ldpc_code_params


def ldpc_bp_decode(llr_vec, ldpc_code_params, decoder_algorithm, n_iters):
    """
    LDPC Decoder using Belief Propagation (BP). If several blocks are provided, they are all decoded at once.

    Parameters
    ----------
    llr_vec : 1D array of float with a length multiple of block length.
        Received codeword LLR values from the channel. They will be clipped in [-500, 500].

    ldpc_code_params : dictionary that at least contains these parameters
        Parameters of the LDPC code as provided by `get_ldpc_code_params`:
            n_vnodes (int) - number of variable nodes.
            n_cnodes (int) - number of check nodes.
            max_vnode_deg (int) - maximal degree of a variable node.
            max_cnode_deg (int) - maximal degree of a check node.
            vnode_adj_list (1D-ndarray of ints) - flatten array so that
                vnode_adj_list.reshape((n_vnodes, max_vnode_deg)) gives for each variable node the adjacent check nodes.
            cnode_adj_list (1D-ndarray of ints) - flatten array so that
                cnode_adj_list.reshape((n_cnodes, max_cnode_deg)) gives for each check node the adjacent variable nodes.
            vnode_cnode_map (1D-ndarray of ints) - flatten array providing the mapping between vnode and cnode indexes.
            cnode_vnode_map (1D-ndarray of ints) - flatten array providing the mapping between vnode and cnode indexes.
            vnode_deg_list (1D-ndarray of ints) - degree of each variable node.
            cnode_deg_list (1D-ndarray of ints) - degree of each check node.

    decoder_algorithm: string
        Specify the decoder algorithm type.
        'SPA' for Sum-Product Algorithm
        'MSA' for Min-Sum Algorithm

    n_iters : int
        Max. number of iterations of decoding to be done.

    Returns
    -------
    dec_word : 1D array or 2D array of 0's and 1's with one block per column.
        The codeword after decoding.

    out_llrs : 1D array or 2D array of float with one block per column.
        LLR values corresponding to the decoded output.
    """

    # Clip LLRs
    llr_vec.clip(-_llr_max, _llr_max, llr_vec)

    # Build parity_check_matrix if required
    if ldpc_code_params.get('parity_check_matrix') is None:
        build_matrix(ldpc_code_params)

    # Initialization
    dec_word = np.signbit(llr_vec)
    out_llrs = llr_vec.copy()
    parity_check_matrix = ldpc_code_params['parity_check_matrix'].astype(float).tocoo()

    for i_start in range(0, llr_vec.size, ldpc_code_params['n_vnodes']):
        i_stop = i_start + ldpc_code_params['n_vnodes']
        message_matrix = parity_check_matrix.multiply(llr_vec[i_start:i_stop])

        # Main loop of Belief Propagation (BP) decoding iterations
        for iter_cnt in range(n_iters):

            # Compute early termination using parity check matrix
            if np.all(ldpc_code_params['parity_check_matrix'].multiply(dec_word[i_start:i_stop]).sum(1) % 2 == 0):
                break

            # Check Node Update
            if decoder_algorithm == 'SPA':
                # Compute incoming messages
                message_matrix.data *= .5
                np.tanh(message_matrix.data, out=message_matrix.data)

                # Runtime Warnings are expected when llr = 0. No warn should be raised as this case are expected.
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Compute product as exponent of the sum of logarithm
                    log2_msg_matrix = message_matrix.astype(complex).copy()
                    np.log2(message_matrix.data.astype(complex), out=log2_msg_matrix.data)
                    msg_products = np.exp2(log2_msg_matrix.sum(1)).real

                    # Compute outgoing messages
                    message_matrix.data = 1 / message_matrix.data
                    message_matrix = message_matrix.multiply(msg_products)
                    message_matrix.data.clip(-1, 1, message_matrix.data)
                    np.arctanh(message_matrix.data, out=message_matrix.data)
                    message_matrix.data *= 2
                    message_matrix.data.clip(-_llr_max, _llr_max, message_matrix.data)

            elif decoder_algorithm == 'MSA':
                message_matrix = message_matrix.tocsr()
                for row_idx in range(message_matrix.shape[0]):
                    begin_row = message_matrix.indptr[row_idx]
                    end_row = message_matrix.indptr[row_idx+1]
                    row_data = message_matrix.data[begin_row:end_row].copy()
                    indexes = np.arange(len(row_data))
                    for j, i in enumerate(range(begin_row, end_row)):
                        other_val = row_data[indexes != j]
                        message_matrix.data[i] = np.sign(other_val).prod() * np.abs(other_val).min()
            else:
                raise NameError('Please input a valid decoder_algorithm string (meanning "SPA" or "MSA").')

            # Variable Node Update
            msg_sum = np.array(message_matrix.sum(0)).squeeze()
            message_matrix.data *= -1
            message_matrix.data += parity_check_matrix.multiply(msg_sum + llr_vec[i_start:i_stop]).data

            out_llrs[i_start:i_stop] = msg_sum + llr_vec[i_start:i_stop]
            np.signbit(out_llrs[i_start:i_stop], out=dec_word[i_start:i_stop])

    # Reformat outputs
    n_blocks = llr_vec.size // ldpc_code_params['n_vnodes']
    dec_word = dec_word.reshape(-1, n_blocks, order='F').squeeze().astype(np.int8)
    out_llrs = out_llrs.reshape(-1, n_blocks, order='F').squeeze()
    return dec_word, out_llrs


def write_ldpc_params(parity_check_matrix, file_path):
    """
    Write parameters from LDPC parity check matrix to a design file.

    The file is structured as followed (examples are available in designs/ldpc/):
        n_vnode n_cnode
        max_vnode_deg max_cnode_deg
        List of the degree of each vnode
        List of the degree of each cnode
        For each vnode (line by line, separated by '\t'): index of the connected cnodes
        For each cnode (line by line, separated by '\t'): index of the connected vnodes

    Parameters
    ----------
    parity_check_matrix : 2D-array of int
        Parity check matrix to save.

    file_path
        File path of the LDPC code design file.
    """
    with open(file_path, 'x') as file:
        file.write('{} {}\n'.format(parity_check_matrix.shape[1], parity_check_matrix.shape[0]))
        file.write('{} {}\n'.format(parity_check_matrix.sum(0).max(), parity_check_matrix.sum(1).max()))

        for deg in parity_check_matrix.sum(0):
            file.write('{} '.format(deg))
        file.write('\n')
        for deg in parity_check_matrix.sum(1):
            file.write('{} '.format(deg))
        file.write('\n')

        for line in parity_check_matrix.T:
            nodes = line.nonzero()[0]
            for node in nodes[:-1]:
                file.write('{}\t'.format(node + 1))
            file.write('{}\n'.format(nodes[-1] + 1))

        for col in parity_check_matrix:
            nodes = col.nonzero()[0]
            for node in nodes[:-1]:
                file.write('{}\t'.format(node + 1))
            file.write('{}\n'.format(nodes[-1] + 1))
        file.write('\n')


def triang_ldpc_systematic_encode(message_bits, ldpc_code_params, pad=True):
    """
    Encode bits using the LDPC code specified. If the  generator matrix is not computed, this function will build it
    and add it to the dictionary. It will also add the parity check matrix.

    This function work only for LDPC specified by a triangular parity check matrix.

    Parameters
    ----------
    message_bits : 1D-array
        Message bit to encode.

    ldpc_code_params : dictionary that at least contains one of these options:
        Option 1: generator matrix is available.
                generator_matrix (2D-array or sparse matrix) - generator matrix of the code.
        Option 2: generator and parity check matrices will be added as sparse matrices.
                n_vnodes (int) - number of variable nodes.
                n_cnodes (int) - number of check nodes.
                max_cnode_deg (int) - maximal degree of a check node.
                cnode_adj_list (1D-ndarray of ints) - flatten array so that
                    cnode_adj_list.reshape((n_cnodes, max_cnode_deg)) gives for each check node the adjacent variable nodes.
                cnode_deg_list (1D-ndarray of ints) - degree of each check node.

    pad : boolean
            Whether to add '0' padding to the message to fit the block length.
            *Default* is True.

    Returns
    -------
    coded_message : 1D-ndarray or 2D-ndarray of int8 depending on the number of blocks
                    Coded message with the systematic part at the beginning.

    Raises
    ------
        ValueError
            If the message length is not a multiple of block length and pad is False.
    """

    if ldpc_code_params.get('generator_matrix') is None:
        build_matrix(ldpc_code_params)

    block_length = ldpc_code_params['generator_matrix'].shape[1]
    modulo = len(message_bits) % block_length
    if modulo:
        if pad:
            message_bits = np.concatenate((message_bits, np.zeros(block_length - modulo, message_bits.dtype)))
        else:
            raise ValueError('Padding is disable but message length is not a multiple of block length.')
    message_bits = message_bits.reshape(block_length, -1, order='F')

    parity_part = ldpc_code_params['generator_matrix'].dot(message_bits) % 2
    return np.vstack((message_bits, parity_part)).squeeze().astype(np.int8)
