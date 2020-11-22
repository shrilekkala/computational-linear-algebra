import numpy as np


def get_Lk(m, lvec):
    """Compute the lower triangular row operation mxm matrix L_k 
    which has ones on the diagonal, and below diagonal entries
    in column k given by lvec (k is inferred from the size of lvec).

    :param m: integer giving the dimensions of L.
    :param lvec: a m-k-1 dimensional numpy array.

    :return Lk: an mxm dimensional numpy array.

    """
    k = m - lvec.size - 1
    Lk = np.eye(m)
    Lk[k+1:,k] = Lk[k+1:,k]  + lvec

    return Lk


def LU_inplace(A):
    """Compute the LU factorisation of A, using the in-place scheme so
    that the strictly lower triangular components of the array contain
    the strictly lower triangular components of L, and the upper
    triangular components of the array contain the upper triangular
    components of U.

    :param A: an mxm-dimensional numpy array

    """
    m = np.shape(A)[0]

    for k in range(m-1):
        A[k+1:, k] = A[k+1:, k] / A[k, k]
        A[k+1:, k+1:] = A[k+1:, k+1:] - np.outer(A[k+1:, k], A[k, k+1:])


def solve_L(L, b):
    """Solve systems Lx_i=b_i for x_i with L lower triangular, i = 1, 2 ... ,k

    :param L: an mxm-dimensional numpy array, assumed lower triangular
    :param b: an mxk-dimensional numpy array, with ith column containing \
    b_i

    :return x: an mxk-dimensional numpy array, with ith column containing \
    the solution x_i

    """
    m = np.shape(b)[0]
    x = np.zeros(b.shape)

    # Implementing the forward substitution algorithm in Page 38 of Chapter 4

    # edge case
    x[0, :] = b[0, :] / L[0, 0]

    # i cycles from 1 to m-1
    for i in range(1, m):
        x[i, :] = (b[i, :] - L[i, :i] @ x[: i, :]) / L[i, i]

    return x


def solve_U(U, b):

    """Solve systems Ux_i=b_i for x_i with U upper triangular, i = 1,2, ...,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing \
    b_i

    :return x: an mxk-dimensional numpy array, with ith column containing \
    the solution x_i

    """
    m = np.shape(b)[0]
    x = np.zeros(b.shape)

    # Using the same algorithm as Ex 3.26 but extending it

    # edge case
    x[m-1, :] = b[m-1, :] / U[m-1,m-1]

    # i cycles from m-2 to 0
    for i in range(m-2, -1, -1):
        x[i, :] = (b[i, :] - U[i, i+1:] @ x[i+1: , :]) / U[i, i]

    return x


def inverse_LU(A):
    """
    Form the inverse of A via LU factorisation.

    :param A: an mxm-dimensional numpy array.

    :return Ainv: an mxm-dimensional numpy array.

    """
    m = np.shape(A)[0]

    # compute the LU factorisation of A in place
    LU_inplace(A)
    i1 = np.tril_indices(m, k=-1)
    L = np.eye(m)
    L[i1] = A[i1]
    U = np.triu(A)

    # find the inverse of L by forward substitution
    # L Linv = I
    Linv = solve_L(L, np.eye(m))

    # find the inverse of A by backward substitution
    # U Ainv = Linv 
    Ainv = solve_U(U, Linv)

    return Ainv