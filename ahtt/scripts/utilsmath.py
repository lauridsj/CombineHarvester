import numpy as np
import numba

@numba.njit(cache=True, nogil=True)
def index_n1(idxn, nbins):
    '''
    translate ND list of indices into an 'unrolled' 1D list of indices
    e.g. 2D -> 1D
    [
    0, 0 -> 0
    0, 1 -> 1
    1, 0 -> 2
    1, 1 -> 3
    ]
    '''
    idx1 = idxn[0]
    for ii in range(1, len(idxn)):
        multiplier = 1
        for jj in range(ii - 1, -1, -1):
            multiplier *= nbins[jj]

        idx1 += idxn[ii] * multiplier

    return idx1

@numba.njit(cache=True, nogil=True)
def index_1d_1n(idx1, dim, nbins):
    '''
    inverse operation of index_n1, for a specific dimension
    '''
    multiplier = 1
    for dd in range(dim - 1, -1, -1):
        multiplier *= nbins[dd]

    return (idx1 // multiplier) % nbins[dim]

@numba.njit(cache=True, nogil=True)
def index_1n(idx1, nbins):
    '''
    as above, but over all dimensions in a go
    '''
    idxn = np.full(len(nbins), -1)
    #idxn = [-1] * len(nbins)
    for iv in range(len(nbins)):
        idxn[iv] = index_1d_1n(idx1, iv, nbins)

    return idxn