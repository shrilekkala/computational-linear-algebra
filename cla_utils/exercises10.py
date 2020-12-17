import numpy as np
import numpy.random as random
import scipy.linalg
# from cla_utils.exercises3 import householder_ls

def arnoldi(A, b, k):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations

    :return Q: an mxk dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper
    Hessenberg matrix
    """
    m = np.size(b)
    Q = np.zeros((m, k+1), dtype = 'complex')
    H = np.zeros((k+1,k), dtype = 'complex')

    Q[:,0] = b / np.linalg.norm(b)

    for n in range(k):
        v = A @ Q[:,n]

        # projection
        H[:n+1, n] = Q[:, :n+1].T.conjugate() @ v
        v = v - Q[:, :n+1] @ H[:n+1, n]

        H[n+1, n] = np.linalg.norm(v)
        Q[:, n+1] = v / np.linalg.norm(v)
    
    return Q, H

def householder(A, kmax=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce
    to upper triangular. If not present, will default to n.

    :return R: an mxn-dimensional numpy array containing the upper
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
        A[:, kmax:][k-1:,:] = A[:, kmax:][k-1:,:] - 2 * np.outer(v_k, v_k) @ A[:, kmax:][k-1:,:]
    return A

def householder_ls_recycle(A, b, Qh = None, Rh = None):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """
    mm, nn = AA.shape
    
    ### construct augmented matrix Ahat
    ###A_hat = np.hstack((A, b[:, None]))

    # use housholder to computer Q and R
    if Qh is not None:
        A_hat = np.hstack((AA, bb[:, None]))

        # apply the last householder reflector to the augmented matrix
        x = A_hat[mm-1: mm+1, mm-1]
        e_1 = np.eye(np.size(x,0))[:,0]
        v_k = np.sign(x[0]) * np.linalg.norm(x) * e_1 + x
        v_k = v_k / np.linalg.norm(v_k)
        A_hat[mm-1: mm+1, mm-1: mm] = A_hat[mm-1: mm+1, mm-1: mm]  - 2 * np.outer(v_k, v_k) @ A_hat[mm-1: mm+1, mm-1: mm]
        A_hat_householder = A_hat
    else:
        # construct augmented matrix Ahat
        A_hat = np.hstack((AA, bb[:, None]))

        A_hat_householder = householder(A_hat, nn)

    # extract R and Q^* b using slice notation
    R = A_hat_householder[:nn, :nn]
    Q_star_b = A_hat_householder[:nn, nn]

    Q = Q_star_b.T.conjugate()
    R = R @ Rh

    # solve the upper triangular system  Rx= Q^* b
    x = scipy.linalg.solve_triangular(R[:, :mm], Q_star_b)

    return x, Q, R

def householder_full_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the full QR factorisation of A.
    :param A: an mxn-dimensional numpy array
    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """
    m, n= A.shape
    
    # create an mxm identity matrix
    I = np.eye(m)

    # construct extended array Ahat
    A_hat = np.hstack((A, I))

    # use housholder to computer Q and R
    A_hat_householder = householder(A_hat, n)

    # extract Q_star and R using slice notation
    R = A_hat_householder[:, :n]
    Q_star = A_hat_householder[:, n:]

    return Q_star.T, R

def extra_householder(Hk, Qhp, k):
    """
    Function that applies one extra Householder rotation for a GMRES iteration
    :inputs: Hk, Qh and k (iteration number)
    :outputs: the new Q and R
    """
    m, n= Hk.shape
    
    Qh = scipy.linalg.block_diag(Qhp, 1)
    H = Qh.T @ Hk
    
    x = H[k: m+1, k]
    I = np.eye(np.size(x,0))

    e_1 = I[:,0]
    v = np.sign(x[0]) * np.linalg.norm(x) * e_1 + x
    F = I - 2 * np.outer(v, v.conjugate()) / (v.T.conjugate() @ v)
    Qhk = scipy.linalg.block_diag(np.eye(k), F)
    
    newR = Qhk @ H
    newQ = Qh @ Qhk.T

    return newQ, newR

def extra_givens(Hk, Qhp, k):
    """
    Function that applies one extra Givens rotation for a GMRES iteration
    :inputs: Hk, Qh and k (iteration number)
    :outputs: the new Q and R
    """    
    if k == 0:
        Qh = np.eye(2)
    else:
        Qh = scipy.linalg.block_diag(Qhp, 1)
    
    A = Qh.T @ Hk
    
    # A[1,0] = 0
    
    theta = np.arctan(A[k+1,k] / A[k,k])
    c = np.cos(theta)
    s = np.sin(theta)
    M = np.array([[c, s],[-s, c]])
    
    A[k: k+2, :] = M @ A[k: k+2, :]
    Qhk = scipy.linalg.block_diag(np.eye(k), M)
    
    newR = A.copy()
    newQ = (Qhk @ Qh.T).T

    return newQ, newR

def GMRES(A, b, maxit, tol, x0=None, return_residual_norms=False, return_residuals=False):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual
    at iteration k
    """
    m = np.size(b)
    
    if x0 is None:
        x0 = b
    
    # normalise the inital vector
    x = x0 / np.linalg.norm(x0)

    # count number of iterations
    k = 0

    # matrix storing the norms of the residuals at each iteration
    rnorms = np.zeros(maxit)
    # list storing the residuals at each iteration
    r = []

    ###
    Qh = None
    Rh = None

    Q = np.zeros((m, maxit+1))
    H = np.zeros((maxit+1,maxit))
    Q[:,0] = x / np.linalg.norm(x)

    while True:
        # Apply step n of Arnoldi
        v = A @ Q[:,k]

        H[:k+1, k] = Q[:, :k+1].T.conjugate() @ v
        v = v - Q[:, :k+1] @ H[:k+1, k]

        H[k+1, k] = np.linalg.norm(v)
        Q[:, k+1] = v / np.linalg.norm(v)

        Hk = H[:(k+1)+1,:(k)+1]
        Qk = Q[:(k+1),:(k+1)]

        # create basis vector e1
        e1 = np.eye((k+1)+1)[:,0]
        
        Qh, Rh = extra_givens(Hk, Qh, k)

        Qh_reduced = Qh[:, :k+1]    
        Rh_reduced = Rh[:k+1, :k+1]
        # Find y by least squares
        y = scipy.linalg.solve_triangular(Rh_reduced, Qh_reduced.conjugate().T @ (np.linalg.norm(b) * e1))
        
        x = Qk @ y

        # update the residuals and residual norms
        r.append(Hk @ y - np.linalg.norm(b) * e1)
        rnorms[k] = np.linalg.norm(r[k])

        k += 1

        # check convergence criteria
        R = rnorms[k-1]
        if R < tol:
            nits = k
            break
        elif k+1 > maxit:
            nits = -1
            break
    
    if return_residual_norms:
        if return_residuals:
            return x, nits, rnorms[:nits+1], r
        else:
            return x, nits, rnorms[:nits+1]
    else:
        if return_residuals:
            return x, nits, r
        else:
            return x, nits


def get_AA100():
    """
    Get the AA100 matrix.

    :return A: a 100x100 numpy array used in exercises 10.
    """
    AA100 = np.fromfile('AA100.dat', sep=' ')
    AA100 = AA100.reshape((100, 100))
    return AA100


def get_BB100():
    """
    Get the BB100 matrix.

    :return B: a 100x100 numpy array used in exercises 10.
    """
    BB100 = np.fromfile('BB100.dat', sep=' ')
    BB100 = BB100.reshape((100, 100))
    return BB100


def get_CC100():
    """
    Get the CC100 matrix.

    :return C: a 100x100 numpy array used in exercises 10.
    """
    CC100 = np.fromfile('CC100.dat', sep=' ')
    CC100 = CC100.reshape((100, 100))
    return CC100

