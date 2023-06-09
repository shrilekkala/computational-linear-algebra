import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from cla_utils.exercises3 import householder_qr_complex

def get_A100():
    """
    Return A100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 50
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    return A


def get_B100():
    """
    Return B100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 50
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A[np.tril_indices(m, -2)] = 0
    return A


def get_C100():
    """
    Return C100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 50
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    return A


def get_D100():
    """
    Return D100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 50
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    A[np.tril_indices(m, -2)] = 0
    A[np.triu_indices(m, 2)] = 0
    return A


def get_A3():
    """
    Return A3 matrix for investigating power iteration.
    
    :return A3: a 3x3 numpy array.
    """

    return np.array([[ 0.68557183+0.46550108j,  0.12934765-0.1622676j ,
                    0.24409518+0.25335939j],
                  [ 0.1531015 +0.66678983j,  0.45112492+0.18206976j,
                    -0.02633966+0.43477693j],
                  [-0.10817164-1.16879196j, -0.18446849+0.03755672j,
                   0.06430325-0.44757084j]])


def get_B3():
    """
    Return B3 matrix for investigating power iteration.

    :return B3: a 3x3 numpy array.
    """
    return np.array([[ 0.46870499+0.37541453j,  0.19115959-0.39233203j,
                    0.12830659+0.12102382j],
                  [ 0.90249603-0.09446345j,  0.51584055+0.84326503j,
                    -0.02582305+0.23259079j],
                  [ 0.75419973-0.52470311j, -0.59173739+0.48075322j,
                    0.51545446-0.21867957j]])


def pow_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either 

    ||r|| < tol where

    r = Ax - lambda*x,

    or the number of iterations exceeds maxit.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence
    of power iterates, instead of just the final iteration. Default is
    False.

    :return x: an m dimensional numpy array containing the final iterate, or
    if store_iterations, an mxmaxit dimensional numpy array containing all
    the iterates.
    :return lambda0: the final eigenvalue.
    """
    m = A.shape[0]

    # count number of iterations
    k = 0

    # matrix storing sequence of iterations
    V = np.zeros((m, maxit+1), dtype = 'complex')
    V[:,0] = x0

    while True:
        w = A @ V[:,k]
        V[:,k+1] = w / np.linalg.norm(w)
        lambda0 = V[:,k+1].conjugate().T @ A @ V[:,k+1]

        # check truncation criteria
        k += 1
        r = A @ V[:,k] - lambda0 * V[:,k]
        if np.linalg.norm(r) < tol:
            break
        elif k+1 > maxit:
            break
    
    if store_iterations:
        x = V[:,1:k+1]
    else:
        x = V[:,k]
    
    return x, lambda0

def ex5_16():
    A3 = get_A3()
    B3 = get_B3()
    v = np.random.randn(3)
    x_a, lambda_a = pow_it(A3, v, tol=1.0e-6, maxit=10000, store_iterations = True)
    x_b, lambda_b = pow_it(B3, v, tol=1.0e-6, maxit=10000, store_iterations = True)
    print("Error in A3 power iteration: " + str(np.linalg.norm(A3@x_a[:,-1]-lambda_a*x_a[:,-1])))
    print("Error in B3 power iteration: " + str(np.linalg.norm(B3@x_b[:,-1]-lambda_b*x_b[:,-1])))
    print("Determinant of A3")
    print(np.linalg.det(A3))
    print("Eigenvalues of B3")
    print(np.linalg.eig(B3)[0])
    return

def inverse_it(A, x0, mu, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence
    of inverse iterates, instead of just the final iteration. Default is
    False.

    :return x: an m dimensional numpy array containing the final iterate, or
    if store_iterations, an mxmaxit dimensional numpy array containing
    all the iterates.
    :return l: a floating point number containing the final eigenvalue
    estimate, or if store_iterations, an m dimensional numpy array containing
    all the iterates.
    """
    m = A.shape[0]
    I = np.eye(m, dtype = 'complex')

    # normalise the inital vector
    x0 = x0 / np.linalg.norm(x0)

    # count number of iterations
    k = 0

    # matrices storing sequences of iterations
    V = np.zeros((m, maxit+1), dtype = 'complex')
    V[:,0] = x0
    L = np.zeros(maxit+1, dtype = 'complex')
    L[0] = mu
    

    while True:
        w = np.linalg.solve(A - mu * I, V[:,k])
        V[:,k+1] = w / np.linalg.norm(w)
        L[k+1] = V[:,k+1].conjugate().T @ A @ V[:,k+1]

        k += 1

        # check truncation criteria
        r = A @ V[:,k] - L[k] * V[:,k]
        if np.linalg.norm(r) < tol:
            break
        elif k+1 > maxit:
            break
    
    if store_iterations:
        x = V[:,1:k+1]
        l = L[1:k+1]
    else:
        x = V[:,k]
        l = L[k]
        
    return x, l

def ex5_18(mu):
    A3 = get_A3()
    B3 = get_B3()
    v = np.random.randn(3)
    e_a, _ = np.linalg.eig(A3)
    e_b, _ = np.linalg.eig(B3)
    x_a, lambda_a = inverse_it(A3, v, mu, tol=1.0e-6, maxit=10000, store_iterations = True)
    x_b, lambda_b = inverse_it(B3, v, mu, tol=1.0e-6, maxit=10000, store_iterations = True)
    print("Error in A3 power iteration: ", np.linalg.norm(A3@x_a[:,-1]-lambda_a[-1]*x_a[:,-1]))
    print("Error in B3 power iteration: ", np.linalg.norm(B3@x_b[:,-1]-lambda_b[-1]*x_b[:,-1]))
    print("Eigenvalues of A3          : ", e_a)
    print("Final lambda_a             : ", lambda_a[-1])
    print("Eigenvalues of B3          : ", e_b)
    print("Final lambda_b             : ", lambda_b[-1])
    return
# ex5_18(0.1)


def rq_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence
    of inverse iterates, instead of just the final iteration. Default is
    False.

    :return x: an m dimensional numpy array containing the final iterate, or
    if store_iterations, an mxmaxit dimensional numpy array containing
    all the iterates.
    :return l: a floating point number containing the final eigenvalue
    estimate, or if store_iterations, an m dimensional numpy array containing
    all the iterates.
    """
    m = A.shape[0]
    I = np.eye(m, dtype = 'complex')

    # normalise the inital vector
    x0 = x0 / np.linalg.norm(x0)

    # count number of iterations
    k = 0

    # matrices storing sequences of iterations
    V = np.zeros((m, maxit+1), dtype = 'complex')
    V[:,0] = x0
    L = np.zeros(maxit+1, dtype = 'complex')
    L[0] = x0.conj().T @ A @ x0

    while True:
        w = np.linalg.solve(A - L[k] * I, V[:,k])
        V[:,k+1] = w / np.linalg.norm(w)
        L[k+1] = V[:,k+1].conjugate().T @ A @ V[:,k+1]

        k += 1

        # check truncation criteria
        r = A @ V[:,k] - L[k] * V[:,k]
        if np.linalg.norm(r) < tol:
            break
        elif k+1 > maxit:
            break
    
    if store_iterations:
        x = V[:,1:k+1]
        l = L[1:k+1]
    else:
        x = V[:,k]
        l = L[k]
        
    return x, l

# create a random Hermitian matrix of dimension m x m
def get_Am(m):
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    return A

def ex5_20(m):
    A = get_Am(m)
    mu = 0
    v = np.random.randn(m)
    _, lambda_inv = inverse_it(A, v, mu, tol=1.0e-10, maxit=10000, store_iterations = True)
    _, lambda_rq = rq_it(A, v, tol=1.0e-10, maxit=10000, store_iterations = True)
    print("Number of iterations for inverse iteration: ", len(lambda_inv))
    print("Number of iterations for r-q iteration    : ", len(lambda_rq))
    #print(lambda_inv)
    #print(lambda_rq)
    return
# ex5_20(1000)

def pure_QR(A, maxit, tol, its=False):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance
    :param its: if True, return number of iterations

    :return Ak: the result
    :return k: number of iterations
    """
    Ak = A.copy()
    m, _ = A.shape
    I = np.eye(m, dtype = 'complex')

    # list storing diagonal elements of each Ak
    A_diagonal = list(np.diag(Ak))

    # counter
    k = 0

    while True:
        # obtain the Q,R decomposition of A implicitly via householder and return the product RQ
        Ak = householder_qr_complex(Ak, RQ_product = True)

        # add the new diagonal of A to the matrix
        A_diagonal.append(np.diag(Ak))

        k += 1

        # check convergence criteria (elementwise convergence of diagonals of A)
        if np.allclose(A_diagonal[-1], A_diagonal[-2], rtol = tol**2):
            break
        elif k+1 > maxit:
            break
    if its:
        return Ak, k
    else:
        return Ak

# pure_QR via inbuilt QR decomposition (for Ex 5.22)
def pure_QR_inbuilt(A, maxit, tol, its=False):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance
    :param its: if True, return number of iterations

    :return Ak: the result
    :return k: number of iterations
    """
    Ak = A.copy()
    m, _ = A.shape
    I = np.eye(m, dtype = 'complex')

    # list storing diagonal elements of each Ak
    A_diagonal = list(np.diag(Ak))

    # counter
    k = 0

    while True:
        # obtain the Q,R decomposition of A implicitly via householder and return the product RQ
        Q, R = np.linalg.qr(Ak)
        Ak = R@Q
        # add the new diagonal of A to the matrix
        A_diagonal.append(np.diag(Ak))

        k += 1
        # check convergence criteria (elementwise convergence of diagonals of A)
        if np.allclose(A_diagonal[-1], A_diagonal[-2], rtol = tol**2):
            break
        elif k+1 > maxit:
            break
    if its:
        return Ak, k
    else:
        return Ak

def ex5_22():
    # Obtain the required matrices
    # random square matrix
    A100 = get_A100()
    # upper triangular
    B100 = get_B100()
    # Hermitian matrix
    C100 = get_C100()
    # Banded Hermitian matrix (middle 3 diagonals)
    D100 = get_D100()

    # Apply pure QR algorithm to each of these
    # using built - in pure QR method as it is faster
    maxiters = 10000
    Ak100, A100_its = pure_QR_inbuilt(A100, maxit=maxiters, tol=1.0e-5, its=True)
    print(A100_its)
    Bk100, B100_its = pure_QR_inbuilt(B100, maxit=maxiters, tol=1.0e-5, its=True)
    print(B100_its)
    Ck100, C100_its = pure_QR_inbuilt(C100, maxit=maxiters, tol=1.0e-5, its=True)
    print(C100_its)
    Dk100, D100_its = pure_QR_inbuilt(D100, maxit=maxiters, tol=1.0e-5, its=True)
    print(D100_its)

    # Visualise matrices A100, B100, C100, D100
    plt.pcolor(np.abs(np.flip(A100,0)))
    plt.title("A100")
    plt.show()
    plt.pcolor(np.abs(np.flip(B100,0)))
    plt.title("B100")
    plt.show()
    plt.pcolor(np.abs(np.flip(C100,0)))
    plt.title("C100")
    plt.show()
    plt.pcolor(np.abs(np.flip(D100,0)))
    plt.title("D100")
    plt.show()

    # Visualise the resulting matrices A100, B100, C100, D100
    plt.pcolor(np.abs(np.flip(Ak100,0)))
    plt.title("A100, k = " + str(A100_its))
    plt.show()
    plt.pcolor(np.abs(np.flip(Bk100,0)))
    plt.title("B100, k = " + str(B100_its))
    plt.show()
    plt.pcolor(np.abs(np.flip(Ck100,0)))
    plt.title("C100, k = " + str(C100_its))
    plt.show()
    plt.pcolor(np.abs(np.flip(Dk100,0)))
    plt.title("D100, k = " + str(D100_its))
    plt.show()

    return()

# ex5_22()