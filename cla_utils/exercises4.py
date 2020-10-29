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
    # generate random matrix A and random vector x
    A = random.randn(m, n)
    x = random.randn(n, 1)

    # check if the inequality is satisfied
    if operator_2_norm(A@x) <= operator_2_norm(A) * operator_2_norm(x):
        print("Inequality satisfied")
    else:
        print("Inequality not satisfied")

# check if the inequality is satisfied for particular m, n
verify_ineq(20, 7)


def verify_thm(l, m, n):
    # generate random matrices A and B
    A = random.randn(l, m)
    B = random.randn(m, n)

    # check if the inequality in the theorem is satisfied
    if operator_2_norm(A@B) <= operator_2_norm(A) * operator_2_norm(B):
        print("Inequality satisfied")
    else:
        print("Inequality not satisfied")

# check if the Theorem is satisfied for particular l, m, n
verify_thm(64, 32, 16)


def cond(A):
    """
    Given a real mxn matrix A, return the condition number in the 2-norm.

    :return A: an mxn-dimensional numpy array

    :param ncond: the condition number
    """
    # compute the eigenvalues of A^* A
    evals = np.linalg.eigvals(A.conjugate().T @ A)

    # find the largest and smallest eigenvalue of A^* A
    lambda_max =  max(evals)
    lambda_min = min(evals)

    # compute the operator norms of A and A inverse
    operator_A = np.sqrt(lambda_max)
    operator_A_inv = np.sqrt(1 / lambda_min)

    # compute the condition number of A
    ncond = operator_A * operator_A_inv
    
    return ncond
