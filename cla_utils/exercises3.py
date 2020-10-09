import numpy as np


def householder(A, kmax=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations.

    Inputs

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.

    Outputs

    :param R: an mxn-dimensional numpy array containing the upper \
    triangular matrix
    """

    m, n = A.shape
    if kmax is None:
        kmax = n

    raise NotImplementedError

    return R


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    Inputs

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    Outputs

    :param x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """

    raise NotImplementedError

    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the QR factorisation of A.

    Inputs

    :param A: an mxn-dimensional numpy array

    Outputs

    :param Q: an mxm-dimensional numpy array
    :param R: an mxn-dimensional numpy array
    """

    raise NotImplementedError

    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    Inputs

    :param A: an mxn-dimensional numpy array
    :param A: an m-dimensional numpy array

    Outputs

    :param x: an n-dimensional numpy array
    """

    raise NotImplementedError

    return x
