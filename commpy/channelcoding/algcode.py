

# Authors: Veeresh Taranalli <veeresht@gmail.com>
# License: BSD 3-Clause

from fractions import gcd
from numpy import array, arange, concatenate, convolve

from commpy.channelcoding.gfields import GF, polymultiply, poly_to_string
from commpy.utilities import dec2bitarray, bitarray2dec

__all__ = ['cyclic_code_genpoly']

def cyclic_code_genpoly(n, k):
    """
    Generate all possible generator polynomials for a (n, k)-cyclic code.

    Parameters
    ----------
    n : int
        Code blocklength of the cyclic code.

    k : int
        Information blocklength of the cyclic code.

    Returns
    -------
    poly_list : 1D ndarray of ints
        A list of generator polynomials (represented as integers) for the (n, k)-cyclic code.

    """


    if n%2 == 0:
        raise ValueError("n cannot be an even number")

    for m in arange(1, 18):
        if (2**m-1)%n == 0:
            break

    x_gf = GF(arange(1, 2**m), m)
    coset_fields = x_gf.cosets()

    coset_leaders = array([])
    minpol_degrees = array([])
    for field in coset_fields:
        coset_leaders = concatenate((coset_leaders, array([field.elements[0]])))
        minpol_degrees = concatenate((minpol_degrees, array([len(field.elements)])))

    y_gf = GF(coset_leaders, m)
    minpol_list = y_gf.minpolys()
    idx_list = arange(1, len(minpol_list))
    poly_list = array([])

    for i in range(1, 2**len(minpol_list)):
        i_array = dec2bitarray(i, len(minpol_list))
        subset_array = minpol_degrees[i_array == 1]
        if int(subset_array.sum()) == (n-k):
            poly_set = minpol_list[i_array == 1]
            gpoly = 1
            for poly in poly_set:
                gpoly_array = dec2bitarray(gpoly, 2**m)
                poly_array = dec2bitarray(poly, 2**m)
                gpoly = bitarray2dec(convolve(gpoly_array, poly_array) % 2)
            poly_list = concatenate((poly_list, array([gpoly])))

    return poly_list.astype(int)


if __name__ == "__main__":
    genpolys = cyclic_code_genpoly(31, 21)
    for poly in genpolys:
        print(poly_to_string(poly))
