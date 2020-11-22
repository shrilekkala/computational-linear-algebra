import numpy as np

def perm(p, i, j):
    """
    For p representing a permutation P, i.e. Px[i] = x[p[i]],
    replace with p representing the permutation P_{i,j}P, where
    P_{i,j} exchanges rows i and j.

    :param p: an m-dimensional numpy array of integers.
    """
    """
    m = p.size
    P = np.zeros((m, m))

    # convert float array to integer array
    p = p.astype(int)


    for i in range(m):
        P[p[i], i] = 1

    
    Pij = np.eye(m)
    Pij[[j, i]] = Pij[[i, j]]
    
    # retrun a vector of indices
    p1 = np.zeros(p)
    p1  Pij @ P
    """

    # swap the ith element and the jth element of the vector of indices p
    p[i], p[j] = p[j], p[i]



def LUP_inplace(A):
    """
    Compute the LUP factorisation of A with partial pivoting, using the
    in-place scheme so that the strictly lower triangular components
    of the array contain the strictly lower triangular components of
    L, and the upper triangular components of the array contain the
    upper triangular components of U.

    :param A: an mxm-dimensional numpy array

    :return p: an m-dimensional integer array describing the permutation \
    i.e. (Px)[i] = x[p[i]]
    """
                     
    raise NotImplementedError


def solve_LUP(A, b):
    """
    Solve Ax=b using LUP factorisation.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an m-dimensional numpy array
    """
                     
    raise NotImplementedError

def det_LUP(A):
    """
    Find the determinant of A using LUP factorisation.

    :param A: an mxm-dimensional numpy array

    :return detA: floating point number, the determinant.
    """
                     
    raise NotImplementedError
