import numpy as np
from numpy import random

def operator_2_norm(A):
    """
    Given a real mxn matrix A, return the operator 2-norm.

    :param A: an mxn-dimensional numpy array

    :return o2norm: the norm
    """

    # compute the largest eigenvalue of A^* A
    lambda_max = max(np.linalg.eigvals(A.conjugate().T @ A))

    # the operator norm of A is the square root of lambda_max
    o2norm = np.sqrt(lambda_max)

    return o2norm

def verify_ineq(m, n):
    A = random.randn(m, n)
    x = random.randn(n, 1)

    # check if the inequality is satisfied
    if operator_2_norm(A@x) <= operator_2_norm(A) * operator_2_norm(x):
        print("Inequality satisfied")
    else:
        print("Inequality not satisfied")

# check if the inequality is satisfied for particular m, n
verify_ineq(20, 7)

def cond(A):
    """
    Given a real mxn matrix A, return the condition number in the 2-norm.

    :return A: an mxn-dimensional numpy array

    :param ncond: the condition number
    """

    raise NotImplementedError

    return ncond
