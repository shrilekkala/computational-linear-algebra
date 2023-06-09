import numpy as np
import numpy.random as random
import timeit

def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """

    # initialise r and u
    r = v
    u = np.zeros(np.size(Q,1),dtype=v.dtype)

    # i cycles from 0 to m-1
    for i in range(0,np.size(Q,1)):

        # Calculate q_i^* v (inner product)
        qi_v = np.inner(Q[:,i].conjugate(),v)
        
        # Store the coefficient in u
        u[i] = qi_v

        # Recursively update r
        r = r - qi_v * Q[:,i]

    return r, u


def solveQ(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """
    # calculate Q^*
    Qstar = np.conjugate(Q).T

    # calculate x
    x = Qstar@b

    return x

# timeable function solveQ
def timeable_solveQ(matrix_size):
    """
    Doing an example with solveQ that we can
    pass to timeit.
    """
    m = matrix_size
    random.seed(1500*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    v = random.randn(m) + 1j*random.randn(m)
    Q, v = np.linalg.qr(A)
    
    x = solveQ(Q, v) #noqa
    return

# timeable np.linalg.solve function
def timeable_numpy_solveQ(matrix_size):
    """
    Doing a matvec example with the builtin numpy matvec so that
    we can pass to timeit.
    """
    m = matrix_size
    random.seed(1500*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    v = random.randn(m) + 1j*random.randn(m)
    Q, v = np.linalg.qr(A)
    
    x = np.linalg.solve(Q, v) #noqa
    return
    
def timeable_solveQ_100():
    timeable_solveQ(100)
    return

def timeable_solveQ_200():
    timeable_solveQ(200)
    return

def timeable_solveQ_400():
    timeable_solveQ(400)
    return

def timeable_numpy_solveQ_100():
    timeable_numpy_solveQ(100)
    return

def timeable_numpy_solveQ_200():
    timeable_numpy_solveQ(200)
    return

def timeable_numpy_solveQ_400():
    timeable_numpy_solveQ(400)
    return

# compare times for the 2 functions for m = 100 , 200 , 400
def time_solveQ():
    """
    Get some timings for the solving Q functions.
    """
    print("Timing for solveQ              m = 100")
    print(timeit.Timer(timeable_solveQ_100).timeit(number=1))
    print("Timing for numpy.linalg.solve, m = 100")
    print(timeit.Timer(timeable_numpy_solveQ_100).timeit(number=1))
    
    print("Timing for solveQ,             m = 200")
    print(timeit.Timer(timeable_solveQ_200).timeit(number=1))
    print("Timing for numpy.linalg.solve, m = 200")
    print(timeit.Timer(timeable_numpy_solveQ_200).timeit(number=1))
    
    print("Timing for solveQ,             m = 400")
    print(timeit.Timer(timeable_solveQ_400).timeit(number=1))
    print("Timing for numpy.linalg.solve, m = 400")
    print(timeit.Timer(timeable_numpy_solveQ_400).timeit(number=1))
    return

def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """
    # Initialise P
    P = np.zeros(np.size(Q,0),dtype=Q.dtype)

    # i cycles from 0 to m-1
    for i in range(0, np.size(Q,1)):
        P  = P + np.outer(Q[:,i],Q[:,i].conjugate())

    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an lxm-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U.
    """
    # Create an orthogonal Projector using the previous function
    P = orthog_proj(V)

    # Perform QR decomposition on P
    QQ, R = np.linalg.qr(P)

    # The first n columns of QQ form an orthonormal basis for V
    # The final m-n columns of QQ form an orthonormal basis for V-complement (null space of V)

    # Return the final m-n columns of QQ
    return QQ[:,np.size(V,1):]


def GS_classical(A):
    """
    Given an mxn matrix A, compute the reduced QR factorisation by classical
    Gram-Schmidt algorithm.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """
    # set m and n from A
    m, n = A.shape

    # initialise Q and R
    R = np.zeros((n, n),dtype=A.dtype)
    Q = A.copy()

    # j cycles from 0 to n-1
    for j in range(0, n):

        # if statement to account for the edge case j = 0
        if j > 0:
            # using slice notation instead of the inner FOR loop
            R[:j-1, j] = Q[:, :j-1].conjugate().T @ Q[:, j]
            Q[:, j] = Q[:, j] - Q[:, :j-1] @ R[:j-1, j]

        # update the matrices R and Q iteratively
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / R[j, j]

    return Q, R

def GS_modified(A):
    """
    Given an mxn matrix A, compute the reduced QR factorisation by modified
    Gram-Schmidt algorithm, producing

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """
    # set m and n from A
    m, n = A.shape

    # initialise Q and R
    R = np.zeros((n, n),dtype=A.dtype)
    Q = A.copy()

    # i cycles from 0 to n-1
    for i in range (0, n):
        R[i, i] = np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i, i]

        # using slice notation instead of the inner FOR loop
        R[i, i+1:] = Q[:, i+1:].conjugate().T @ Q[:,i]
        Q[:, i+1:] = Q[:, i+1:] - np.outer(Q[:, i], R[i, i+1:].T)
    
    return Q, R

def test_mutual_orthogonality():
    m = 600
    n = 450
    
    random.seed(1705*m + 1248*n)

    # artificially construct a basis with good linear independence
    A = random.randn(m, m) + 1j*random.randn(m, m)
    U, _ = np.linalg.qr(A)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    V, _ = np.linalg.qr(A)
    
    # change the parameter to test using large values in matrices
    D = np.diag(1.0 + 50*random.rand(m))
    A = np.dot(U, np.dot(D, V))
    A = A[:, 0:n]

    Qc, Rc = GS_classical(A)
    Qm, Rm = GS_modified(A)
    
    ZeroN = np.zeros((n, n),dtype='complex')
    
    # print(Qc.conjugate().T @ Qm - Qm.conjugate().T @ Qc)
    
    # Norm of Qm^* Qc - Qc^* Qm (as a measure of mutual orthogonality)
    ### print(np.linalg.norm(Qc.conjugate().T @ Qm - Qm.conjugate().T @ Qc - ZeroN))
    
    return Qc, Rc, Qm, Rm

test_mutual_orthogonality()


def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """
    m, n = A.shape

    # initialise R
    R = np.eye(n, dtype=A.dtype)

    # edge case 
    R[k-1, k-1] = 1 / np.linalg.norm(A[:, k-1])

    R[k-1, k-1:n] = - (A[:, k-1].conjugate().T /np.linalg.norm(A[:, k-1]) @ A[:,k-1:n]) * R[k-1, k-1]

    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    R = np.eye(n, dtype=A.dtype)
    for i in range(1,n+1):
        Rk = GS_modified_get_R(A, i)
        A = A @ Rk
        R = R @ Rk
    R = np.linalg.inv(R)
    return A, R