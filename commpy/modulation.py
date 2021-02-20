# Authors: CommPy contributors
# License: BSD 3-Clause

"""
==================================================
Modulation Demodulation (:mod:`commpy.modulation`)
==================================================

.. autosummary::
   :toctree: generated/

   PSKModem             -- Phase Shift Keying (PSK) Modem.
   QAMModem             -- Quadrature Amplitude Modulation (QAM) Modem.
   ofdm_tx              -- OFDM Transmit Signal Generation
   ofdm_rx              -- OFDM Receive Signal Processing
   mimo_ml              -- MIMO Maximum Likelihood (ML) Detection.
   kbest                -- MIMO K-best Schnorr-Euchner Detection.
   best_first_detector  -- MIMO Best-First Detection.
   bit_lvl_repr         -- Bit Level Representation.
   max_log_approx       -- Max-Log Approximation.

"""
from bisect import insort

import matplotlib.pyplot as plt
from numpy import arange, array, zeros, pi, sqrt, log2, argmin, \
    hstack, repeat, tile, dot, shape, concatenate, exp, \
    log, vectorize, empty, eye, kron, inf, full, abs, newaxis, minimum, clip, fromiter
from numpy.fft import fft, ifft
from numpy.linalg import qr, norm
from sympy.combinatorics.graycode import GrayCode

from commpy.utilities import bitarray2dec, dec2bitarray, signal_power

__all__ = ['PSKModem', 'QAMModem', 'ofdm_tx', 'ofdm_rx', 'mimo_ml', 'kbest', 'best_first_detector',
           'bit_lvl_repr', 'max_log_approx']


class Modem:

    """ Creates a custom Modem object.

        Parameters
        ----------
        constellation : array-like with a length which is a power of 2
                        Constellation of the custom modem

        Attributes
        ----------
        constellation : 1D-ndarray of complex
                        Modem constellation. If changed, the length of the new constellation must be a power of 2.

        Es            : float
                        Average energy per symbols.

        m             : integer
                        Constellation length.

        num_bits_symb : integer
                        Number of bits per symbol.

        Raises
        ------
        ValueError
                        If the constellation is changed to an array-like with length that is not a power of 2.
        """

    def __init__(self, constellation, reorder_as_gray=True):
        """ Creates a custom Modem object. """

        if reorder_as_gray:
            m = log2(len(constellation))
            gray_code_sequence = GrayCode(m).generate_gray()
            gray_code_sequence_array = fromiter((int(g, 2) for g in gray_code_sequence), int, len(constellation))
            self.constellation = array(constellation)[gray_code_sequence_array.argsort()]
        else:
            self.constellation = constellation

    def modulate(self, input_bits):
        """ Modulate (map) an array of bits to constellation symbols.

        Parameters
        ----------
        input_bits : 1D ndarray of ints
            Inputs bits to be modulated (mapped).

        Returns
        -------
        baseband_symbols : 1D ndarray of complex floats
            Modulated complex symbols.

        """
        mapfunc = vectorize(lambda i:
                            self._constellation[bitarray2dec(input_bits[i:i + self.num_bits_symbol])])

        baseband_symbols = mapfunc(arange(0, len(input_bits), self.num_bits_symbol))

        return baseband_symbols

    def demodulate(self, input_symbols, demod_type, noise_var=0):
        """ Demodulate (map) a set of constellation symbols to corresponding bits.

        Parameters
        ----------
        input_symbols : 1D ndarray of complex floats
            Input symbols to be demodulated.

        demod_type : string
            'hard' for hard decision output (bits)
            'soft' for soft decision output (LLRs)

        noise_var : float
            AWGN variance. Needs to be specified only if demod_type is 'soft'

        Returns
        -------
        demod_bits : 1D ndarray of ints
            Corresponding demodulated bits.

        """
        if demod_type == 'hard':
            index_list = abs(input_symbols - self._constellation[:, None]).argmin(0)
            demod_bits = dec2bitarray(index_list, self.num_bits_symbol)

        elif demod_type == 'soft':
            demod_bits = zeros(len(input_symbols) * self.num_bits_symbol)
            for i in arange(len(input_symbols)):
                current_symbol = input_symbols[i]
                for bit_index in arange(self.num_bits_symbol):
                    llr_num = 0
                    llr_den = 0
                    for bit_value, symbol in enumerate(self._constellation):
                        if (bit_value >> bit_index) & 1:
                            llr_num += exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                        else:
                            llr_den += exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                    demod_bits[i * self.num_bits_symbol + self.num_bits_symbol - 1 - bit_index] = log(llr_num / llr_den)
        else:
            raise ValueError('demod_type must be "hard" or "soft"')

        return demod_bits

    def plot_constellation(self):
        """ Plot the constellation """
        plt.scatter(self.constellation.real, self.constellation.imag)

        for symb in self.constellation:
            plt.text(symb.real + .2, symb.imag, self.demodulate(symb, 'hard'))

        plt.title('Constellation')
        plt.grid()
        plt.show()

    @property
    def constellation(self):
        """ Constellation of the modem. """
        return self._constellation

    @constellation.setter
    def constellation(self, value):
        # Check value input
        num_bits_symbol = log2(len(value))
        if num_bits_symbol != int(num_bits_symbol):
            raise ValueError('Constellation length must be a power of 2.')

        # Set constellation as an array
        self._constellation = array(value)

        # Update other attributes
        self.Es = signal_power(self.constellation)
        self.m = self._constellation.size
        self.num_bits_symbol = int(num_bits_symbol)


class PSKModem(Modem):
    """ Creates a Phase Shift Keying (PSK) Modem object.

        Parameters
        ----------
        m : int
            Size of the PSK constellation.

        Attributes
        ----------
        constellation : 1D-ndarray of complex
                        Modem constellation. If changed, the length of the new constellation must be a power of 2.

        Es            : float
                        Average energy per symbols.

        m             : integer
                        Constellation length.

        num_bits_symb : integer
                        Number of bits per symbol.

        Raises
        ------
        ValueError
                        If the constellation is changed to an array-like with length that is not a power of 2.
        """

    def __init__(self, m):
        """ Creates a Phase Shift Keying (PSK) Modem object. """

        num_bits_symbol = log2(m)
        if num_bits_symbol != int(num_bits_symbol):
            raise ValueError('Constellation length must be a power of 2.')

        super().__init__(exp(1j * arange(0, 2 * pi, 2 * pi / m)))


class QAMModem(Modem):
    """ Creates a Quadrature Amplitude Modulation (QAM) Modem object.

        Parameters
        ----------
        m : int
            Size of the PSK constellation.

        Attributes
        ----------
        constellation : 1D-ndarray of complex
                        Modem constellation. If changed, the length of the new constellation must be a power of 2.

        Es            : float
                        Average energy per symbols.

        m             : integer
                        Constellation length.

        num_bits_symb : integer
                        Number of bits per symbol.

        Raises
        ------
        ValueError
                        If the constellation is changed to an array-like with length that is not a power of 2.
                        If the parameter m would lead to an non-square QAM during initialization.
    """

    def __init__(self, m):
        """ Creates a Quadrature Amplitude Modulation (QAM) Modem object.

        Parameters
        ----------
        m : int
            Size of the QAM constellation. Must lead to a square QAM (ie sqrt(m) is an integer).

        Raises
        ------
        ValueError
                        If m would lead to an non-square QAM.
        """

        num_symb_pam = sqrt(m)
        if num_symb_pam != int(num_symb_pam):
            raise ValueError('m must lead to a square QAM.')

        pam = arange(-num_symb_pam + 1, num_symb_pam, 2)
        constellation = tile(hstack((pam, pam[::-1])), int(num_symb_pam) // 2) * 1j + pam.repeat(num_symb_pam)
        super().__init__(constellation)


def ofdm_tx(x, nfft, nsc, cp_length):
    """ OFDM Transmit signal generation """

    nfft = float(nfft)
    nsc = float(nsc)
    cp_length = float(cp_length)
    ofdm_tx_signal = array([])

    for i in range(0, shape(x)[1]):
        symbols = x[:, i]
        ofdm_sym_freq = zeros(nfft, dtype=complex)
        ofdm_sym_freq[1:(nsc / 2) + 1] = symbols[nsc / 2:]
        ofdm_sym_freq[-(nsc / 2):] = symbols[0:nsc / 2]
        ofdm_sym_time = ifft(ofdm_sym_freq)
        cp = ofdm_sym_time[-cp_length:]
        ofdm_tx_signal = concatenate((ofdm_tx_signal, cp, ofdm_sym_time))

    return ofdm_tx_signal


def ofdm_rx(y, nfft, nsc, cp_length):
    """ OFDM Receive Signal Processing """

    num_ofdm_symbols = int(len(y) / (nfft + cp_length))
    x_hat = zeros([nsc, num_ofdm_symbols], dtype=complex)

    for i in range(0, num_ofdm_symbols):
        ofdm_symbol = y[i * nfft + (i + 1) * cp_length:(i + 1) * (nfft + cp_length)]
        symbols_freq = fft(ofdm_symbol)
        x_hat[:, i] = concatenate((symbols_freq[-nsc / 2:], symbols_freq[1:(nsc / 2) + 1]))

    return x_hat


def mimo_ml(y, h, constellation):
    """ MIMO ML Detection.

    parameters
    ----------
    y : 1D ndarray of complex floats
        Received complex symbols (shape: num_receive_antennas x 1)

    h : 2D ndarray of complex floats
        Channel Matrix (shape: num_receive_antennas x num_transmit_antennas)

    constellation : 1D ndarray of complex floats
        Constellation used to modulate the symbols

    """
    _, n = h.shape
    m = len(constellation)
    x_ideal = empty((n, pow(m, n)), complex)
    for i in range(0, n):
        x_ideal[i] = repeat(tile(constellation, pow(m, i)), pow(m, n - i - 1))
    min_idx = argmin(norm(y[:, None] - dot(h, x_ideal), axis=0))
    x_r = x_ideal[:, min_idx]

    return x_r


def kbest(y, h, constellation, K, noise_var=0, output_type='hard', demode=None):
    """ MIMO K-best Schnorr-Euchner Detection.

    Reference: Zhan Guo and P. Nilsson, 'Algorithm and implementation of the K-best sphere decoding for MIMO detection',
        IEEE Journal on Selected Areas in Communications, vol. 24, no. 3, pp. 491-503, Mar. 2006.

    Parameters
    ----------
    y : 1D ndarray
        Received complex symbols (length: num_receive_antennas)

    h : 2D ndarray
        Channel Matrix (shape: num_receive_antennas x num_transmit_antennas)

    constellation : 1D ndarray of floats
        Constellation used to modulate the symbols

    K : positive integer
        Number of candidates kept at each step

    noise_var : positive float
        Noise variance.
        *Default* value is 0.

    output_type : str
        'hard': hard output i.e. output is a binary word
        'soft': soft output i.e. output is a vector of Log-Likelihood Ratios.
        *Default* value is 'hard'

    demode : function with prototype binary_word = demode(point)
        Function that provide the binary word corresponding to a symbol vector.

    Returns
    -------
    x : 1D ndarray of constellation points or of Log-Likelihood Ratios.
        Detected vector (length: num_receive_antennas).

    raises
    ------
    ValueError
                If h has more columns than rows.
                If output_type is something else than 'hard' or 'soft'
    """
    nb_tx, nb_rx = h.shape
    if nb_rx > nb_tx:
        raise ValueError('h has more columns than rows')

    # QR decomposition
    q, r = qr(h)
    yt = q.conj().T.dot(y)

    # Initialization
    m = len(constellation)
    nb_can = 1

    if isinstance(constellation[0], complex):
        const_type = complex
    else:
        const_type = float
    X = empty((nb_rx, K * m), dtype=const_type)  # Set of current candidates
    d = tile(yt[:, None], (1, K * m))  # Corresponding distance vector
    d_tot = zeros(K * m, dtype=float)  # Corresponding total distance
    hyp = empty(K * m, dtype=const_type)  # Hypothesis vector

    # Processing
    for coor in range(nb_rx - 1, -1, -1):
        nb_hyp = nb_can * m

        # Copy best candidates m times
        X[:, :nb_hyp] = tile(X[:, :nb_can], (1, m))
        d[:, :nb_hyp] = tile(d[:, :nb_can], (1, m))
        d_tot[:nb_hyp] = tile(d_tot[:nb_can], (1, m))

        # Make hypothesis
        hyp[:nb_hyp] = repeat(constellation, nb_can)
        X[coor, :nb_hyp] = hyp[:nb_hyp]
        d[coor, :nb_hyp] -= r[coor, coor] * hyp[:nb_hyp]
        d_tot[:nb_hyp] += abs(d[coor, :nb_hyp]) ** 2

        # Select best candidates
        argsort = d_tot[:nb_hyp].argsort()
        nb_can = min(nb_hyp, K)  # Update number of candidate

        # Update accordingly
        X[:, :nb_can] = X[:, argsort[:nb_can]]
        d[:, :nb_can] = d[:, argsort[:nb_can]]
        d[:coor, :nb_can] -= r[:coor, coor, None] * hyp[argsort[:nb_can]]
        d_tot[:nb_can] = d_tot[argsort[:nb_can]]

    if output_type == 'hard':
        return X[:, 0]
    elif output_type == 'soft':
        return max_log_approx(y, h, noise_var, X[:, :nb_can], demode)
    else:
        raise ValueError('output_type must be "hard" or "soft"')


def best_first_detector(y, h, constellation, stack_size, noise_var, demode, llr_max):
    """ MIMO Best-First Detection.

    Reference: G. He, X. Zhang, et Z. Liang, "Algorithm and Architecture of an Efficient MIMO Detector With Cross-Level
     Parallel Tree-Search", IEEE Transactions on Very Large Scale Integration (VLSI) Systems, 2019


    Parameters
    ----------
    y : 1D ndarray
        Received complex symbols (length: num_receive_antennas)

    h : 2D ndarray
        Channel Matrix (shape: num_receive_antennas x num_transmit_antennas)

    constellation : 1D ndarray of floats
        Constellation used to modulate the symbols

    stack_size : tuple of integers
        Size of each stack (length: num_transmit_antennas - 1)

    noise_var : positive float
        Noise variance.
        *Default* value is 0.

    demode : function with prototype binary_word = demode(point)
        Function that provide the binary word corresponding to a symbol vector.

    llr_max : float
        Max value for LLR clipping

    Returns
    -------
    x : 1D ndarray of Log-Likelihood Ratios.
        Detected vector (length: num_receive_antennas).
    """

    class _Node:
        """ Helper data model that implements __lt__ (aka '<') as required to use bisect.insort. """

        def __init__(self, symb_vectors, partial_metrics):
            """
            Recursive initializer that build a sequence of siblings.
            Inputs are assumed to be ordered based on metric
            """
            if len(partial_metrics) == 1:
                # There is one node to build
                self.symb_vector = symb_vectors.reshape(-1)  # Insure that self.symb_vector is a 1d-ndarray
                self.partial_metric = partial_metrics[0]
                self.best_sibling = None
            else:
                # Recursive call to build several nodes
                self.symb_vector = symb_vectors[:, 0].reshape(-1)  # Insure that self.symb_vector is a 1d-ndarray
                self.partial_metric = partial_metrics[0]
                self.best_sibling = _Node(symb_vectors[:, 1:], partial_metrics[1:])

        def __lt__(self, other):
            return self.partial_metric < other.partial_metric

        def expand(self, yt, r, constellation):
            """ Build all children and return the best one. constellation must be a numpy ndarray."""
            # Construct children's symbol vector
            child_size = self.symb_vector.size + 1
            children_symb_vectors = empty((child_size, constellation.size), constellation.dtype)
            children_symb_vectors[1:] = self.symb_vector[:, newaxis]
            children_symb_vectors[0] = constellation

            # Compute children's partial metric and sort
            children_metric = abs(yt[-child_size] - r[-child_size, -child_size:].dot(children_symb_vectors)) ** 2
            children_metric += self.partial_metric
            ordering = children_metric.argsort()

            # Build children and return the best one
            return _Node(children_symb_vectors[:, ordering], children_metric[ordering])

    # Extract information from arguments
    nb_tx, nb_rx = h.shape
    constellation = array(constellation)
    m = constellation.size
    modulation_order = int(log2(m))

    # QR decomposition
    q, r = qr(h)
    yt = q.conj().T.dot(y)

    # Initialisation
    map_metric = inf
    map_bit_vector = None
    counter_hyp_metric = full((nb_tx, modulation_order), inf)
    stacks = tuple([] for _ in range(nb_tx))

    # Start process by adding the best root's child in the last stack
    stacks[-1].append(_Node(empty(0, constellation.dtype), array(0, float, ndmin=1)).expand(yt, r, constellation))

    # While there is at least one non-empty stack (exempt the first one)
    while any(stacks[1:]):
        # Node processing
        for idx_next_stack in range(len(stacks) - 1):
            try:
                idx_this_stack = idx_next_stack + 1
                best_node = stacks[idx_this_stack].pop(0)

                # Update search radius
                if map_bit_vector is None:
                    radius = inf  # No leaf has been reached yet so we keep all nodes
                else:
                    bit_vector = demode(best_node.symb_vector).reshape(-1, modulation_order)
                    bit_vector[bit_vector == 0] = -1

                    # Select the counter hyp metrics that could be affected by this node. Details: eq. (14)-(16) in [1].
                    try:
                        a2 = counter_hyp_metric[idx_this_stack:][map_bit_vector[idx_this_stack:] != bit_vector].max()
                    except ValueError:
                        a2 = inf  # NumPy cannot compute max on an empty matrix
                    radius = max(counter_hyp_metric[:idx_this_stack].max(), a2)

                # Process best sibling
                if best_node.best_sibling is not None and best_node.best_sibling.partial_metric <= radius:
                    insort(stacks[idx_this_stack], best_node.best_sibling)

                # Process children
                best_child = best_node.expand(yt, r, constellation)
                if best_child.partial_metric <= radius:
                    insort(stacks[idx_next_stack], best_child)
            except IndexError:  # Raised when popping an empty stack
                pass

        # LLR update if there is a new leaf
        if stacks[0]:
            if stacks[0][0].partial_metric < map_metric:
                minimum(counter_hyp_metric, map_metric, out=counter_hyp_metric)
                map_metric = stacks[0][0].partial_metric
                map_bit_vector = demode(stacks[0][0].symb_vector).reshape(-1, modulation_order)
                map_bit_vector[map_bit_vector == 0] = -1
            else:
                minimum(counter_hyp_metric, stacks[0][0].partial_metric, out=counter_hyp_metric)
            clip(counter_hyp_metric, map_metric - llr_max, map_metric + llr_max, counter_hyp_metric)

        # Trimming stack according to requested max stack size
        del stacks[0][0:]  # there is no stack for the leafs
        for idx_next_stack in range(len(stacks) - 1):
            del stacks[idx_next_stack + 1][stack_size[idx_next_stack]:]

    return ((map_metric - counter_hyp_metric) * map_bit_vector).reshape(-1)


def bit_lvl_repr(H, w):
    """ Bit-level representation of matrix H with weights w.

    parameters
    ----------
    H   :   2D ndarray (shape : nb_rx, nb_tx)
            Channel Matrix.

    w   :   1D ndarray of complex (length : beta)
            Bit level representation weights. The length must be even.

    return
    ------
    A : 2D nbarray (shape : nb_rx, nb_tx*beta)
        Channel matrix adapted to the bit-level representation.

    raises
    ------
    ValueError
                    If beta (the length of w) is not even)
    """
    beta = len(w)
    if beta % 2 == 0:
        m, n = H.shape
        In = eye(n, n)
        kr = kron(In, w)
        return dot(H, kr)
    else:
        raise ValueError('Beta (length of w) must be even.')


def max_log_approx(y, h, noise_var, pts_list, demode):
    """ Max-log demode

    parameters
    ----------
    y : 1D ndarray
        Received symbol vector (length: num_receive_antennas)

    h : 2D ndarray
        Channel Matrix (shape: num_receive_antennas x num_transmit_antennas)

    noise_var : positive float
        Noise variance

    pts_list : 2D ndarray of constellation points
        Set of points to compute max log approximation (points are column-wise).
        (shape: num_receive_antennas x num_points)

    demode : function with prototype binary_word = demode(point)
        Function that provide the binary word corresponding to a symbol vector.

    return
    ------
    LLR : 1D ndarray of floats
        Log-Likelihood Ratio for each bit (same length as the return of decode)
    """
    # Decode all pts
    nb_pts = pts_list.shape[1]
    bits = demode(pts_list.reshape(-1, order='F')).reshape(nb_pts, -1)  # Set of binary words (one word by row)

    # Prepare LLR computation
    nb_bits = bits.shape[1]
    LLR = empty(nb_bits)

    # Loop for each bit
    for k in range(nb_bits):
        # Select pts based on the k-th bit in the corresponding word
        pts0 = pts_list.compress(bits[:, k] == 0, axis=1)
        pts1 = pts_list.compress(bits[:, k] == 1, axis=1)

        # Compute the norms and add inf to handle empty set of points
        norms0 = hstack((norm(y[:, None] - h.dot(pts0), axis=0) ** 2, inf))
        norms1 = hstack((norm(y[:, None] - h.dot(pts1), axis=0) ** 2, inf))

        # Compute LLR
        LLR[k] = min(norms0) - min(norms1)

    return -LLR / (2 * noise_var)
