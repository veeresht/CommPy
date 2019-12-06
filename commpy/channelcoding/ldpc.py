# Authors: CommPy contributors
# License: BSD 3-Clause

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splg

__all__ = ['build_matrix', 'get_ldpc_code_params', 'ldpc_bp_decode', 'write_ldpc_params',
           'triang_ldpc_systematic_encode']


def build_matrix(ldpc_code_params):
    """
    Build the parity check matrix from parameters dictionary and add the result in this dictionary.

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
    """
    n_cnodes = ldpc_code_params['n_cnodes']
    cnode_deg_list = ldpc_code_params['cnode_deg_list']
    cnode_adj_list = ldpc_code_params['cnode_adj_list'].reshape((n_cnodes, ldpc_code_params['max_cnode_deg']))

    parity_check_matrix = sp.lil_matrix((n_cnodes, ldpc_code_params['n_vnodes']), dtype=np.int8)
    for cnode_idx in range(n_cnodes):
        parity_check_matrix[cnode_idx, cnode_adj_list[cnode_idx, :cnode_deg_list[cnode_idx]]] = 1

    ldpc_code_params['parity_check_matrix'] = parity_check_matrix.tocsc()


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


def sum_product_update(cnode_idx, cnode_adj_list, cnode_deg_list, cnode_msgs,
                       vnode_msgs, cnode_vnode_map, max_cnode_deg, max_vnode_deg):

    start_idx = cnode_idx*max_cnode_deg
    offset = cnode_deg_list[cnode_idx]
    vnode_list = cnode_adj_list[start_idx:start_idx+offset]
    vnode_list_msgs_tanh = np.tanh(vnode_msgs[vnode_list*max_vnode_deg +
                                   cnode_vnode_map[start_idx:start_idx+offset]]/2.0)
    msg_prod = np.prod(vnode_list_msgs_tanh)

    # Compute messages on outgoing edges using the incoming message product
    cnode_msgs[start_idx:start_idx+offset]= 2.0*np.arctanh(msg_prod/vnode_list_msgs_tanh)


def min_sum_update(cnode_idx, cnode_adj_list, cnode_deg_list, cnode_msgs,
                   vnode_msgs, cnode_vnode_map, max_cnode_deg, max_vnode_deg):

    start_idx = cnode_idx*max_cnode_deg
    offset = cnode_deg_list[cnode_idx]
    vnode_list = cnode_adj_list[start_idx:start_idx+offset]
    vnode_list_msgs = vnode_msgs[vnode_list*max_vnode_deg + cnode_vnode_map[start_idx:start_idx+offset]]
    vnode_list_msgs = np.ma.array(vnode_list_msgs, mask=False)

    # Compute messages on outgoing edges using the incoming messages
    for i in range(start_idx, start_idx+offset):
        vnode_list_msgs.mask[i-start_idx] = True
        cnode_msgs[i] = np.prod(np.sign(vnode_list_msgs))*np.min(np.abs(vnode_list_msgs))
        vnode_list_msgs.mask[i-start_idx] = False


def ldpc_bp_decode(llr_vec, ldpc_code_params, decoder_algorithm, n_iters):
    """
    LDPC Decoder using Belief Propagation (BP).

    Parameters
    ----------
    llr_vec : 1D array of float
        Received codeword LLR values from the channel.

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
    dec_word : 1D array of 0's and 1's
        The codeword after decoding.

    out_llrs : 1D array of float
        LLR values corresponding to the decoded output.
    """

    n_cnodes = ldpc_code_params['n_cnodes']
    n_vnodes = ldpc_code_params['n_vnodes']
    max_cnode_deg = ldpc_code_params['max_cnode_deg']
    max_vnode_deg = ldpc_code_params['max_vnode_deg']
    cnode_adj_list = ldpc_code_params['cnode_adj_list']
    cnode_vnode_map = ldpc_code_params['cnode_vnode_map']
    vnode_adj_list = ldpc_code_params['vnode_adj_list']
    vnode_cnode_map = ldpc_code_params['vnode_cnode_map']
    cnode_deg_list = ldpc_code_params['cnode_deg_list']
    vnode_deg_list = ldpc_code_params['vnode_deg_list']

    dec_word = np.zeros(n_vnodes, int)
    out_llrs = np.zeros(n_vnodes, int)

    cnode_msgs = np.zeros(n_cnodes*max_cnode_deg)
    vnode_msgs = np.zeros(n_vnodes*max_vnode_deg)

    if decoder_algorithm == 'SPA':
        check_node_update = sum_product_update
    elif decoder_algorithm == 'MSA':
        check_node_update = min_sum_update
    else:
        raise NameError('Please input a valid decoder_algorithm string (meanning "SPA" or "MSA").')

    # Initialize vnode messages with the LLR values received
    for vnode_idx in range(n_vnodes):
        start_idx = vnode_idx*max_vnode_deg
        offset = vnode_deg_list[vnode_idx]
        vnode_msgs[start_idx : start_idx+offset] = llr_vec[vnode_idx]

    # Main loop of Belief Propagation (BP) decoding iterations
    for iter_cnt in range(n_iters):

        continue_flag = 0

        # Check Node Update
        for cnode_idx in range(n_cnodes):

            check_node_update(cnode_idx, cnode_adj_list, cnode_deg_list, cnode_msgs,
                              vnode_msgs, cnode_vnode_map, max_cnode_deg, max_vnode_deg)

        # Variable Node Update
        for vnode_idx in range(n_vnodes):

            # Compute sum of all incoming messages at the variable node
            start_idx = vnode_idx*max_vnode_deg
            offset = vnode_deg_list[vnode_idx]
            cnode_list = vnode_adj_list[start_idx:start_idx+offset]
            cnode_list_msgs = cnode_msgs[cnode_list*max_cnode_deg + vnode_cnode_map[start_idx:start_idx+offset]]
            msg_sum = np.sum(cnode_list_msgs)

            # Compute messages on outgoing edges using the incoming message sum (LLRs are clipped in [-38, 38])
            np.clip(llr_vec[vnode_idx] + msg_sum - cnode_list_msgs, -38, 38, vnode_msgs[start_idx:start_idx+offset])

            # Update output LLRs and decoded word
            out_llrs[vnode_idx] = llr_vec[vnode_idx] + msg_sum
            if out_llrs[vnode_idx] > 0:
                dec_word[vnode_idx] = 0
            else:
                dec_word[vnode_idx] = 1

        # Compute early termination using parity check matrix
        for cnode_idx in range(n_cnodes):
            p_sum = 0
            for i in range(cnode_deg_list[cnode_idx]):
                p_sum ^= dec_word[cnode_adj_list[cnode_idx*max_cnode_deg + i]]

            if p_sum != 0:
                continue_flag = 1
                break

        # Stop iterations
        if continue_flag == 0:
            break

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
    Encode bits using the LDPC code specified. If the parity check matrix and/or the generator matrix are not computed,
    this function will build the missing one(s) and add them to the dictionary.

    This function work only for LDPC specified by a triangular parity check matrix.

    Parameters
    ----------
    message_bits : 1D-array
        Message bit to encode.

    ldpc_code_params : dictionary that at least contains on of these options:
        Option 1: generator matrix is available.
            generator_matrix (2D-array or sparse matrix) - generator matrix of the code.
        Option 2: parity check matrix is available, the generator matrix will be added as a CSR sparse matrix.
            parity_check_matrix (sparse matrix) - parity check matrix of the code.
        Option 3: generator and parity check matrices will be added as sparse matrices of integers.
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
    coded_message : 1D-ndarray or 2D-ndarray depending on the number of blocks
        Coded message with the systematic part at the beginning.

    Raises
    ------
        ValueError
            If the message length is not a multiple of block length and pad is False.
    """

    if ldpc_code_params.get('generator_matrix') is None:
        if ldpc_code_params.get('parity_check_matrix') is None:
            build_matrix(ldpc_code_params)

        parity_check_matrix = ldpc_code_params['parity_check_matrix']
        block_length = parity_check_matrix.shape[0]

        systematic_part = parity_check_matrix[:, -block_length:]
        parity_part = parity_check_matrix[:, :-block_length]
        ldpc_code_params['generator_matrix'] = splg.inv(systematic_part).dot(parity_part).tocsr()

    block_length = ldpc_code_params['generator_matrix'].shape[1]
    modulo = len(message_bits) % block_length
    if modulo:
        if pad:
            message_bits = np.concatenate((message_bits, np.zeros(block_length - modulo, message_bits.dtype)))
        else:
            raise ValueError('Padding is disable but message length is not a multiple of block length.')
    message_bits = message_bits.reshape(block_length, -1, order='F')

    parity_part = ldpc_code_params['generator_matrix'].dot(message_bits) % 2
    return np.vstack((message_bits, parity_part)).squeeze()
