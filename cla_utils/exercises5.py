import numpy as np
from numpy import random
import cla_utils

# from exercises3 import householder_solve
# from exercises4 import cond


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

###m=100
###backward_stability_householder(m)

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
    x[m-1] = b[m-1] / R[m-1, m-1]

    # Iteratively find the other elements of x
    # i cycles from m-2 to 0
    for i in range (m-2, -1, -1):
        x[i] = (b[i] - np.dot(R[i, i+1:], x[i+1: ])) / R[i, i]
    
    return x


def back_stab_solve_R(m):
    """
    Verify backward stability for back substitution for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        # create random matrices R and x
        A = random.randn(m, m)
        Q, R = np.linalg.qr(A)

        x = random.randn(m)
        
        # Find exact b
        b = R @ x

        # obtain x~ by the back subsitution algorithm
        x_tilde = solve_R(R,b)

        print("Backward error is : ", np.linalg.norm(x-x_tilde) / np.linalg.norm(x))
    return

###m=100
###back_stab_solve_R(m)


def back_stab_householder_solve(m):
    """
    Verify backward stability for the householder algorithm
    for solving Ax=b for an m dimensional square system.

    :param m: the matrix dimension parameter.
    """
    # Generate random m x m matrix A and vector x
    A = random.randn(m, m)
    x = random.randn(m, 1)

    # Find exact b
    b = A @ x

    # Obtain x~ by the householder_solve function (Ex 3)
    x_tilde = cla_utils.householder_solve(A,b)

    # Obtain the condition number for this problem (also the condition number of A)
    kappa = cla_utils.cond(A)

    print("(Householder) Backward error is : ", np.linalg.norm(x-x_tilde) / np.linalg.norm(x))
    print("The condition number is", kappa)
    print("Backward error / condition number is", (np.linalg.norm(x-x_tilde) / ((np.linalg.norm(x)) * kappa)))
    return

###m=100
###back_stab_householder_solve(m)
