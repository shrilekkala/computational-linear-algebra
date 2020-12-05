import numpy as np
from numpy import random
import matplotlib.pyplot as plt

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
    m = np.shape(A)[0]

    for k in range (m-2):
        x = A[k+1:, k]
        v = np.sign(x[0]) * np.linalg.norm(x) * np.eye(m-k-1)[:,0] + x
        v = v / np.linalg.norm(v)
        
        # transformation equivalent to left multplication
        A[k+1:, k+1:] = A[k+1:, k+1:] - 2 * np.outer(v, v.conjugate()) @ A[k+1:, k+1:]

        # exploiting the fact that we know where zeros are to be expected
        A[k+1, k] = A[k+1, k] - 2 * v[0] * v.conjugate().T @ A[k+1:, k]
        A[k+2:, k] = np.zeros(m-k-2)

        # transformation equivalent to right multplication
        A[k:, k+1:] = A[k:, k+1:] - 2 * A[k:, k+1:] @ np.outer(v, v.conjugate())
        

def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array
    
    :return Q: an mxm numpy array
    """
    m = np.shape(A)[0]
    Q = np.eye(m, dtype = 'complex')

    for k in range (m-2):
        x = A[k+1:, k]
        v = np.sign(x[0]) * np.linalg.norm(x) * np.eye(m-k-1)[:,0] + x
        v = v / np.linalg.norm(v)

        # transformation equivalent to left multplication (all columns)
        A[k+1:, :] = A[k+1:, :] - 2 * np.outer(v, v.conjugate()) @ A[k+1:, :]

        # transformation equivalent to right multplication (all rows)
        A[:, k+1:] = A[:, k+1:] - 2 * A[:, k+1:] @ np.outer(v, v.conjugate())
    
        # construct Q by the implicit multiplication procedure
        Q[:, k+1:] = Q[:, k+1:] - 2 * Q[:, k+1:] @ np.outer(v, v.conjugate())

    return Q

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
    # reduce A to upper Hessenberg form
    Q = hessenbergQ(A)

    # find the eigenvectors of the Hessenberg matrix
    V = hessenberg_ev(A)

    return Q @ V

def rayleigh_pert(A, v1, lambda1, epsilon):
    e1 = np.eye((len(v1)))[:,0]
    x = v1 + e1 * epsilon

    # compute the rayleigh quotient of the perturbed vector
    rq = x.T @ A @ x / (x.T @ x)

    err = np.abs(lambda1-rq)
    return err

def ex5_13():
    random.seed(123)
    m = 500
    A = np.random.randn(m,m)
    A = A + A.T

    # return the eigenvalues and normalised eigenvectors
    D, V = np.linalg.eig(A)
    lambda1 = D[0]
    v1 = V[:,0]

    N = 100
    eps = np.linspace(0, 1, num=N)
    err = np.zeros(N)

    for i in range(N):
        err[i] = rayleigh_pert(A, v1, lambda1, eps[i])

    plt.loglog(eps[1:], err[1:])
    plt.loglog(eps[1:], 50 * eps[1:]**2)
    plt.show()

ex5_13()