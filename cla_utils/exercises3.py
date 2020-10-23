import numpy as np
import scipy.linalg

def householder(A, kmax=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.

    :return R: an mxn-dimensional numpy array containing the upper \
    triangular matrix
    """

    m, n = A.shape

    if kmax is None:
        kmax = n

    # k cycles from 1 to n
    for k in range(1, kmax+1):

        x = A[k-1: m+1, k-1]

        # initialise e_1 of length m-k+1
        e_1 = np.eye(np.size(x,0))[:,0]


        # householder algorithm
        v_k = np.sign(x[0]) * np.linalg.norm(x) * e_1 + x
        v_k = v_k / np.linalg.norm(v_k)

        A[k-1: m+1, k-1: kmax] = A[k-1: m+1, k-1: kmax]  - 2 * np.outer(v_k, v_k) @ A[k-1: m+1, k-1: kmax]

        # for case when kmax =/= n, compute Q*b in place of b
        A[:, m:][k-1: m+1,:] = A[:, m:][k-1: m+1,:] - 2 * np.outer(v_k, v_k) @ A[:, m:][k-1: m+1,:]
    return A


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """
    m, k= b.shape

    # construct extended array Ahat
    A_hat = np.hstack((A, b))

    # pass A hat to householder function with kmax = m
    R = householder(A_hat, m)
    print(np.shape(R), "R")

    # initialise x
    x = np.zeros((m,k))

    x = scipy.linalg.solve_triangular(R[:, :m], R[:, m:])

    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """

    raise NotImplementedError

    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """

    raise NotImplementedError

    return x
