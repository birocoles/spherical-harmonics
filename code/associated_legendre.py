import numpy as np
from numba import jit


def fully_norm_Plm(colatitude, lmax, method='MFC-1'):
    '''
    Compute the a vector with all fully-normalized associated Legendre functions
    up to a maximum degree 'lmax' at a given spherical 'colatitude' (in degrees).
    The values are stored in an 1D array having 'lmax*(lmax+1)//2 + lmax + 1'
    elements. The element 'l*(l+1)//2 + m' of this array contains the
    associated Legendre function with degree 'l' and order 'm'.

    The code uses numba for speed up computations.

    Parameters
    ----------
    colatitude : float
        Colatitude (in dregrees) at whicn the fully-normalized associated
        Legendre functions will be calculated.

    lmax : int
        Maximum degree of the fully-normalized associated Legendre functions
        to be computed.

    method : {'MFC-1', 'MFC-2', 'MFR'}
        'MFC-1', 'MFC-2' and 'MFR' define the methods 'first Modified Forward
        Column', 'second Modified Forward Column' and 'Modified Forward Row'
        presented by Holmes and Featherstone (2002) to compute
        the fully-normalized associated Legendre functions. Default is 'MFC-1'.

    Returns
    -------
    Plm : array 1D
        Vector containing the computed fully-normalized associated Legendre
        functions.

    References
    ----------
    Holmes, S., Featherstone, W. A unified approach to the Clenshaw summation
    and the recursive computation of very high degree and order normalized
    associated Legendre functions. Journal of Geodesy 76, 279–299 (2002).
    https://doi.org/10.1007/s00190-002-0216-2
    '''

    # check input
    assert np.isscalar(colatitude), 'colatitude must be a scalar'
    assert isinstance(lmax, int), 'lmax must be an integer'
    assert lmax >= 0, 'lmax must be greater than or equal to 0'
    assert method in ['MFC-1', 'MFC-2', 'MFR'], 'method must be MFC-1, MFC-2 \
or MFR'

    # create a dictionary of methods
    methods = {
        'MFC-1': first_modified_forward_column,
        'MFC-2': second_modified_forward_column,
        'MFR': modified_forward_row,
    }

    # transform colatitude from degrees to radians and compute its sine/cosine
    colatitude_rad = np.deg2rad(colatitude)
    sine = np.sin(colatitude_rad)
    cosine = np.cos(colatitude_rad)

    # return analytic values for lmax up to 1
    if lmax == 0:
        return 1
    if lmax == 1:
        Plm = np.zeros(3)
        Plm[0] = 1.
        Plm[1] = cosine*np.sqrt(3)
        Plm[2] = sine*np.sqrt(3)
        return Plm
    if lmax > 1:
        Plm = np.zeros(lmax*(lmax+1)//2 + lmax + 1)
        # define terms depending on P00 and P11, respectively
        Plm[0] = 1e-280
        Plm[2] = 1e-280*np.sqrt(3)

    # compute the scaled sectoral ratios 10^-280 x Pll/(sine^l) for l > 1
    # defined by the sectoral fully-normalized associated Legendre
    # functions Pll
    sectoral_terms(cosine, lmax, Plm)

    # compute the scaled non-sectoral ratios 10^-280 x Plm/(sine^m) for l > 1
    # defined by the non-sectoral fully-normalized associated Legendre
    # functions Plm
    methods[method](cosine, lmax, Plm)

    # remove the scaling factors
    unscaling(Plm, lmax, sine)

    return Plm


@jit(nopython=True)
def sectoral_terms(cosine, lmax, scaled_Plm):
    '''
    Compute the scaled sectoral ratios 10^-280 x Pll/(sine^l) for l > 1 by
    using the recursive algorithm defined by Holmes and Featherstone
    (2002, eq. 28).

    The scaled sectoral ratios are the elements 0, 2, 5, 9, 14, ... , lmax of
    the array 'scaled_Plm'.
    '''

    # indices of the terms defined by P11 and P22, respectively
    index_l_1_l_1 = 2
    index_ll = 5

    for l in range(2, lmax+1):

        # recursion relation for l = m > 1
        factor = np.sqrt((2*l + 1)/(2*l))
        scaled_Plm[index_ll] = factor*scaled_Plm[index_l_1_l_1]

        # update index_l_1_l_1
        index_l_1_l_1 += l + 1

        # update index_ll
        index_ll += l + 2

    return scaled_Plm


@jit(nopython=True)
def first_modified_forward_column(cosine, lmax, scaled_Plm):
    '''
    Compute the scaled non-sectoral ratios 10^-280 x Plm/(sine^m) by using the
    recursive algorithm defined by Holmes and Featherstone (2002, eq. 32).
    The algorithm computes terms of constant order 'm' and sequentially
    increasing degree 'l'.
    '''

    # indices of the terms defined by P00 and P10, respectively
    seed_l_1_m = 0
    seed_lm = 1

    # compute terms 10^-280 x Plm/(sine^m) for l > m
    for m in range(lmax):

        # starting indices
        index_l_1_m = seed_l_1_m
        index_lm = seed_lm

        # recursion relation to compute the term lm from l-1m, where l = m + 1
        scaled_Plm[index_lm] = _a_lm(m+1, m)*cosine*scaled_Plm[index_l_1_m]

        # update indices
        index_l_2_m = index_l_1_m
        index_l_1_m = index_lm
        index_lm += m + 2

        for l in range(m+2, lmax+1):

            # recursion relation to compute the term lm from terms l-1m and l-2m
            scaled_Plm[index_lm] = _a_lm(l, m)*cosine*scaled_Plm[index_l_1_m]
            scaled_Plm[index_lm] -= _b_lm(l, m)*scaled_Plm[index_l_2_m]

            # update indices
            index_l_2_m = index_l_1_m
            index_l_1_m = index_lm
            index_lm += l + 1

        # update seeds
        seed_l_1_m += m + 2
        seed_lm += m + 3

    return scaled_Plm


@jit(nopython=True)
def second_modified_forward_column(cosine, lmax, scaled_Plm):

    return scaled_Plm


@jit(nopython=True)
def modified_forward_row(cosine, lmax, scaled_Plm):

    return scaled_Plm


@jit(nopython=True)
def _a_lm(l, m):
    '''
    Compute the coefficient a_lm (Holmes and Featherstone, 2002, eq. 12)
    with degree l and order m.
    '''
    a_lm = (2*l - 1)*(2*l + 1)
    a_lm /= (l - m)*(l + m)
    a_lm = np.sqrt(a_lm)
    return a_lm


@jit(nopython=True)
def _b_lm(l, m):
    '''
    Compute the coefficient b_lm (Holmes and Featherstone, 2002, eq. 12)
    with degree l and order m.
    '''
    b_lm = (2*l + 1)*(l + m - 1)*(l - m - 1)
    b_lm /= (l - m)*(l + m)*(2*l - 3)
    b_lm = np.sqrt(b_lm)
    return b_lm


@jit(nopython=True)
def unscaling(Plm, lmax, sine):
    '''
    Remove the scaling factor from the computed fully-normalized associated
    Legendre functions.

    Parameters
    ----------
    Plm : array 1D
        Vector containing the computed scaled fully-normalized associated
        Legendre functions.

    lmax : int
        Maximum degree of the fully-normalized associated Legendre functions
        to be computed.

    sine : float
        Sine of the spherical colatitude at which the functions are computed.

    Returns
    -------
    Plm : array 1D
        Vector containing the unscaled fully-normalized associated Legendre
        functions.
    '''

    # remove the scale factor 10^⁻280
    Plm *= 1e280

    # index of the term defined by P11
    seed_ll = 2

    sine_m = sine

    for m in range(1, lmax+1):

        # starting index
        index = seed_ll

        # remove scale factor sine^m
        Plm[index] *= sine_m

        # update index
        index += m + 1

        for l in range(m+1, lmax+1):

            # remove scale factor sine^m
            Plm[index] *= sine_m

            # update index
            index += l + 1

        # update seed
        seed_ll += m + 2

        # update sine_m
        sine_m *= sine

    return Plm


def fully_norm_analytical(colatitude):
    '''
    Compute the fully-normalized associated Legedre polynomials up to degree
    l = 4 at an given spherical colatitude by using analytical expressions.

    Parameters
    ----------
    colatitude : float
        Colatitude (in dregrees) at whicn the fully-normalized associated
        Legendre functions will be calculated.

    Returns
    -------
    Plm: numpy array 1D
        Vector containing the fully-normalized associated Legendre functions
        P00, P10, P11, P20, P21, P22, P30, P31, P32, P33, P40, P41, P42, P43
        and P44 computed at the given spherical colatitude.
    '''

    colat = np.deg2rad(colatitude)
    cosine = np.cos(colat)
    sine = np.sin(colat)
    cosine2 = cosine*cosine
    sine2 = sine*sine

    P00 = _fully_normalization_factor(0,0)*1
    P10 = _fully_normalization_factor(1,0)*cosine
    P11 = _fully_normalization_factor(1,1)*sine
    P20 = _fully_normalization_factor(2,0)*0.5*(3*cosine2 - 1)
    P21 = _fully_normalization_factor(2,1)*3*sine*cosine
    P22 = _fully_normalization_factor(2,2)*3*sine2
    P30 = _fully_normalization_factor(3,0)*0.5*cosine*(5*cosine2 - 3)
    P31 = _fully_normalization_factor(3,1)*1.5*(5*cosine2 - 1)*sine
    P32 = _fully_normalization_factor(3,2)*15*cosine*sine2
    P33 = _fully_normalization_factor(3,3)*15*sine2*sine
    P40 = _fully_normalization_factor(4,0)*0.125*(35*cosine2*cosine2 - 30*cosine2 + 3)
    P41 = _fully_normalization_factor(4,1)*2.5*(7*cosine2*cosine -3*cosine)*sine
    P42 = _fully_normalization_factor(4,2)*7.5*(7*cosine2 - 1)*sine2
    P43 = _fully_normalization_factor(4,3)*105*cosine*sine2*sine
    P44 = _fully_normalization_factor(4,4)*105*sine2*sine2

    Plm = [P00,
           P10, P11,
           P20, P21, P22,
           P30, P31, P32, P33,
           P40, P41, P42, P43, P44]

    Plm = np.array(Plm)

    return Plm


def _fully_normalization_factor(l, m):
    '''
    Given a degree l and an order m, this routine computes the
    normalization factor used to compute the fully-normalized
    associated Legendre functions.
    '''
    if m == 0:
        k = 1
    else:
        k = 2

    l_minus_m_factorial = np.math.factorial(l - m)
    l_plus_m_factorial = np.math.factorial(l + m)
    normalization_factor = k*(2*l + 1)*l_minus_m_factorial/l_plus_m_factorial
    normalization_factor = np.sqrt(normalization_factor)

    return normalization_factor
