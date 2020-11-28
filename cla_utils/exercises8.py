import numpy as np

def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """
    m = np.shape(A)[0]

    # Householder reflector algorithm for first row only
    x = A[:, 0]
    
    v = np.sign(x[0]) * np.linalg.norm(x) * np.eye(m)[:,0] + x
    v = v / np.linalg.norm(v)

    # Apply transformation equivalent to left multplication by Q1 to A
    A = A  - 2 * np.outer(v, v.conjugate()) @ A 

    # Apply the transformation equivalent to right multiplication by Q1^* to A
    A1 = A  - 2 * A @ np.outer(v, v.conjugate())

    return A1

def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """

    raise NotImplementedError


def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array
    
    :return Q: an mxm numpy array
    """

    raise NotImplementedError

def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvalues and eigenvectors.

    :param H: an mxm numpy array

    :return ee: an m dimensional numpy array containing the eigenvalues of H
    :return V: an mxm numpy array whose columns are the eigenvectors of H
    """
    m, n = H.shape
    assert(m==n)
    assert(np.linalg.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V


def ev(A):
    """
    Given a matrix A, return the eigenvectors of A. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """

    raise NotImplementedError