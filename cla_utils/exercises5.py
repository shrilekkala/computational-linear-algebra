import numpy as np
from numpy import random

def randomQ(m):
    """
    Produce a random orthogonal mxm matrix.

    :param m: the matrix dimension parameter.

    :return Q: the mxm numpy array containing the orthogonal matrix.
    """
    Q, R = np.linalg.qr(random.randn(m, m))

    return Q


def randomR(m):
    """
    Produce a random upper triangular mxm matrix.

    :param m: the matrix dimension parameter.

    :return R: the mxm numpy array containing the upper triangular matrix.
    """

    A = random.randn(m, m)
    return np.triu(A)


def backward_stability_householder(m):
    """
    Verify backward stability for QR factorisation using Householder for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        Q1 = randomQ(m)
        R1 = randomR(m)

        A = Q1 @ R1

        Q2, R2 = np.linalg.qr(A)

        print("Norm of Q2 - Q1 is : ", np.linalg.norm(Q2-Q1))
        print("Norm of R2 - R1 is : ", np.linalg.norm(R2-R1))
        print("Norm of A - Q1R2 is : ", np.linalg.norm(A-Q2@R2))
    
    return

m=100
backward_stability_householder(m)

def solve_R(R, b):
    """
    Solve the system Rx=b where R is an mxm upper triangular matrix
    and b is an m dimensional vector.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :param x: an m-dimensional numpy array
    """
    m = np.size(b)

    x = np.zeros(m)
    # Edge case
    x[m-1] = b[m-1] / R[m-1,m-1]

    # Iteratively find the other elements of x
    # i cycles from m-1 to 0
    for i in range (m-2, -1, -1):
        x[i] = (b[i] - np.dot(R[i,i+1:m],x[i+1:m+1])) / R[i,i]
    
    return x


def back_stab_solve_R(m):
    """
    Verify backward stability for back substitution for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        A = random.randn(m, m)
        R = np.triu(A)

        raise NotImplementedError


def back_stab_householder_solve(m):
    """
    Verify backward stability for the householder algorithm
    for solving Ax=b for an m dimensional square system.

    :param m: the matrix dimension parameter.
    """
    raise NotImplementedError
