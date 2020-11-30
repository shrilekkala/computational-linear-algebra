import numpy as np
import scipy.linalg
from cla_utils.exercises6 import solve_L, solve_U

def perm(p, i, j):
    """
    For p representing a permutation P, i.e. Px[i] = x[p[i]],
    replace with p representing the permutation P_{i,j}P, where
    P_{i,j} exchanges rows i and j.

    :param p: an m-dimensional numpy array of integers.
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

    :return p: an m-dimensional integer array describing the permutation
    i.e. (Px)[i] = x[p[i]]
    """
    m = np.shape(A)[0]
    p = np.arange(m)

    for k in range(m-1):

        # select the index of the element with largest magnitude in A[k:, k]
        i = np.absolute(A[k: , k]).argmax(axis = 0)
        i = i + k

        # swap elements k:m of columns i and k
        A[[i, k], k: ] = A[[k, i], k: ]

        # swap elements 1:k-1 of columns i and k
        A[[i, k], :k] = A[[k, i], :k]

        # swap representation of P with representation of Pik @ P as in Ex 4.7
        perm(p, i, k)

        # Gaussian elimination "in-place" algorithm
        A[k+1:, k] = A[k+1:, k] / A[k, k]
        A[k+1:, k+1:] = A[k+1:, k+1:] - np.outer(A[k+1:, k], A[k, k+1:])                 

    return p

def solve_LUP(A, b):
    """
    Solve Ax=b using LUP factorisation.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an m-dimensional numpy array
    """
    m = b.size

    # compute the LUP factorisation of A in place
    p = LUP_inplace(A)
    i1 = np.tril_indices(m, k=-1)
    L = np.eye(m)
    L[i1] = A[i1]
    U = np.triu(A)

    # permute b to Pb
    Pb = b[p]
    
    # find Ux by forward substitution
    # (Note to self: Here Pb is adjusted since solve_L requires 2d arrays)
    Ux = solve_L(A, np.stack((Pb, np.ones(m)), axis = 1), True)

    # find x by backward substitution 
    # (Note to self: can also use solve_U from exercises 5 but need to adjust function to work with 1d b)
    x = solve_U(U, Ux)
    
    return x[:, 0]


def LUP_inplace_n(A):
    """
    Same as LUP_inplace, except return n instead of p

    :return n: the number of row exchanges performed in the algorithm
    """
    m = np.shape(A)[0]
    n = 0

    for k in range(m-1):

        # select the index of the element with largest magnitude in A[k:, k]
        i = np.absolute(A[k: , k]).argmax(axis = 0)
        i = i + k

        # increase n by 1 if a row exchange is made
        if i != k:
            n = n + 1

        # swap elements k:m of columns i and k
        A[[i, k], k: ] = A[[k, i], k: ]

        # swap elements 1:k-1 of columns i and k
        A[[i, k], :k] = A[[k, i], :k]

        # Gaussian elimination "in-place" algorithm
        A[k+1:, k] = A[k+1:, k] / A[k, k]
        A[k+1:, k+1:] = A[k+1:, k+1:] - np.outer(A[k+1:, k], A[k, k+1:])                 

    return n


def det_LUP(A):
    """
    Find the determinant of A using LUP factorisation.

    :param A: an mxm-dimensional numpy array

    :return detA: floating point number, the determinant.
    """
    # compute the LUP factorisation of A in place
    n = LUP_inplace_n(A)
    
    # determinant of P is the (-1) to the power of the number of row exchanges of P
    detP = (- 1) ** n

    # determinant of L is the product of 1s
    detL = 1

    # determinant of U is the product of the diagonal elements of U
    detU = A.diagonal().prod()          

    detA = detL * detU / detP

    return detA

def ex4_13(m):
    A0 = 3 * np.eye(m)
    A0[0, :] = np.ones(m)
    A0[:, 0] = np.arange(1, m+1)
    A = A0 * 1.0

    # In place LU decomposition without pivoting
    for k in range(m-1):
        A[k+1:, k] = A[k+1:, k] / A[k, k]
        A[k+1:, k+1:] = A[k+1:, k+1:] - np.outer(A[k+1:, k], A[k, k+1:])

    i1 = np.tril_indices(m, k=-1)
    L = np.eye(m)
    L[i1] = A[i1]
    U = np.triu(A)

    print("Number of non-zero entries in L: " + str(np.count_nonzero(L)))
    print("Number of non-zero entries in U: " + str(np.count_nonzero(U)))

    return L, U
