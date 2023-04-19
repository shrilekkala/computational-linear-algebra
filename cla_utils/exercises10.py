import numpy as np
import matplotlib.pyplot as plt
from cla_utils.exercises5 import solve_R

def arnoldi(A, b, k):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations

    :return Q: an mx(k+1) dimensional numpy array containing the orthonormal basis
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

def block_diag(A,B):
    """
    Function that takes square matrices or numbers A, B as input
    and returns the block diagonal matrix = diag(A,B)
    """
    if type(A) == np.ndarray:
        m = A.shape[0]
    else:
        m = 1
        
    if type(B) == np.ndarray:
        n = B.shape[0]
    else:
        n = 1
        
    # dimension of new matrix
    k = m + n
    
    C = np.zeros((k,k))
    C[:m,:m] = A
    C[m:,m:] = B
    
    return C

def extra_householder(Hk, Qhp, k):
    """
    Function that applies one extra Householder rotation for a GMRES iteration
    :inputs: Hk, Qh and k (iteration number)
    :outputs: the new Q and R
    """
    m, _ = Hk.shape
    
    Qh = block_diag(Qhp, 1)
    H = Qh.T @ Hk
    
    x = H[k: m+1, k]
    I = np.eye(np.size(x,0))

    e_1 = I[:,0]
    v = np.sign(x[0]) * np.linalg.norm(x) * e_1 + x
    F = I - 2 * np.outer(v, v.conjugate()) / (v.T.conjugate() @ v)
    Qhk = block_diag(np.eye(k), F)
    
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
        Qh = block_diag(Qhp, 1)
    
    A = Qh.T @ Hk
        
    theta = np.arctan(A[k+1,k] / A[k,k])
    c = np.cos(theta)
    s = np.sin(theta)
    M = np.array([[c, s],[-s, c]])
    
    A[k: k+2, :] = M @ A[k: k+2, :]
    Qhk = block_diag(np.eye(k), M)
    
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
        Qk = Q[:,:(k+1)]

        # create basis vector e1
        e1 = np.eye((k+1)+1)[:,0]
        
        # Obtain the QR factorisation of Hk via Givens rotations
        Qh, Rh = extra_givens(Hk, Qh, k)

        # Find y by least squares
        Qh_reduced = Qh[:, :k+1]    
        Rh_reduced = Rh[:k+1, :k+1]
        # back substitution
        y = solve_R(Rh_reduced, Qh_reduced.conjugate().T @ (np.linalg.norm(b) * e1))
        
        # update the solution
        x = Qk @ y

        # update the residuals and residual norms
        r.append(Hk @ y - np.linalg.norm(b) * e1)
        rnorms[k] = np.linalg.norm(r[k])

        # increment the iteration counter
        k += 1

        # check convergence criteria
        R = rnorms[k-1]
        if R < tol:
            nits = k
            break
        elif k+1 > maxit:
            nits = -1
            break
    
    # return the variables required
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

def ex6_5():
    # get matrices
    A = get_AA100()
    B = get_BB100()
    C = get_CC100()

    # get eigenvalues
    A_ev = np.linalg.eig(A)[0]
    B_ev = np.linalg.eig(B)[0]
    C_ev = np.linalg.eig(C)[0]

    # visualise eigenvalues
    for i in ((A_ev, "A"), (B_ev, "B"), (C_ev, "C")):
        plt.title("Eigenvalues of " + i[1])
        plt.scatter(np.arange(100)+1, i[0], marker='x')
        plt.show()

    # obtain random v
    v = np.random.randn(100)

    # perform GMRES
    A_x, A_nits, A_rnorms, A_r = GMRES(A, v.copy(), maxit=1000, tol=1.0e-3, return_residual_norms=True, return_residuals=True)
    B_x, B_nits, B_rnorms, B_r = GMRES(B, v.copy(), maxit=1000, tol=1.0e-3, return_residual_norms=True, return_residuals=True)
    C_x, C_nits, C_rnorms, C_r = GMRES(C, v.copy(), maxit=1000, tol=1.0e-3, return_residual_norms=True, return_residuals=True)

    # print number of iterations for each matrix
    for i in ((A_nits, "A"), (B_nits, "B"), (C_nits, "C")):
        print(i[1] + " converges in: " + str(i[0]) + " iterations")

    # plot the residual norms
    plt.title("Residual Norms")
    plt.plot(A_rnorms, label="A", ls='--')
    plt.plot(B_rnorms, label="B", ls='--')
    plt.plot(C_rnorms, label="C", ls='--')
    plt.legend()
    plt.show()

    # plot eigenvalues of A, B and C as roots on the x axis
    plt.scatter(A_ev, np.zeros(100), marker='x')
    plt.show()
    plt.scatter(B_ev, np.zeros(100), marker='x')
    plt.scatter(C_ev, np.zeros(100), marker='x')
    plt.show()

# ex6_5()