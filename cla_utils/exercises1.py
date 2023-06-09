import numpy as np
import timeit
import numpy.random as random

# pre-construct a matrix in the namespace to use in tests
random.seed(1651)
A0 = random.randn(500, 500)
x0 = random.randn(500)

def basic_matvec(A, x):
    """
    Elementary matrix-vector multiplication.

    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    returns an m-dimensional numpy array which is the product of A with x

    This should be implemented using a double loop over the entries of A

    :return b: m-dimensional numpy array
    """
    # initialise vector b to be a vector of length m
    b = np.zeros(np.size(A,0))

    # i cycles through 1 to m
    for i in range(1,np.size(A,0)+1):
        sum = 0

        # j cycles through 1 to n
        for j in range(1,len(x)+1):

            # adds the product of a_ij and x_j into the sum
            sum += A[i-1][j-1] * x[j-1]

        # update the value of b_i
        b[i-1] = sum
    
    return b
            

def column_matvec(A, x):
    """
    Matrix-vector multiplication using the representation of the product
    Ax as linear combinations of the columns of A, using the entries in 
    x as coefficients.


    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    :return b: an m-dimensional numpy array which is the product of A with x

    This should be implemented using a single loop over the entries of x
    """
    # initialise vector b to be a vector of length m
    b = np.zeros(np.size(A,0))
    
    # j cycles through 1 to n
    for j in range(1,len(x)+1):
        
        # using slice notation to obtain the jth column of A
        b += x[j-1] * A[:,j-1]
    
    # return the final value of b
    return(b)


def timeable_basic_matvec():
    """
    Doing a matvec example with the basic_matvec that we can
    pass to timeit.
    """
    m,n = 300, 500
    random.seed(2020*m + 1066*n)
    A0 = random.randn(m, n)
    x0 = random.randn(n)

    b = basic_matvec(A0, x0) # noqa


def timeable_column_matvec():
    """
    Doing a matvec example with the column_matvec that we can
    pass to timeit.
    """
    m,n = 30, 50
    random.seed(2020*m + 1066*n)
    A0 = random.randn(m, n)
    x0 = random.randn(n)

    b = column_matvec(A0, x0) # noqa


def timeable_numpy_matvec():
    """
    Doing a matvec example with the builtin numpy matvec so that
    we can pass to timeit.
    """
    m,n = 30, 50
    random.seed(2020*m + 1066*n)
    A0 = random.randn(m, n)
    x0 = random.randn(n)

    b = A0.dot(x0) # noqa


def time_matvecs():
    """
    Get some timings for matvecs.
    """

    print("Timing for basic_matvec")
    print(timeit.Timer(timeable_basic_matvec).timeit(number=1))
    print("Timing for column_matvec")
    print(timeit.Timer(timeable_column_matvec).timeit(number=1))
    print("Timing for numpy matvec")
    print(timeit.Timer(timeable_numpy_matvec).timeit(number=1))

### time_matvecs()

def rank2(u1, u2, v1, v2):
    """
    Return the rank2 matrix A = u1*v2^* + u2*v2^*.

    :param u1: m-dimensional numpy array
    :param u1: m-dimensional numpy array
    :param v1: n-dimensional numpy array
    :param v2: n-dimensional numpy array
    """
    # create a m x 2 matrix B consisting of u_1 and u_2 stacked together
    B = np.vstack((u1,u2)).T
    
    # create a 2 x n matrix C consisting of the conjugates of v_1 and v_2 stacked together
    C = np.vstack((v1,v2)).conjugate()

    # A will be the matrix product of B x C
    A = B.dot(C)
    
    # Check the rank of A
    print("The rank of A is:", np.linalg.matrix_rank(A))

    return A


def rank1pert_inv(u, v):
    """
    Return the inverse of the matrix A = I + uv^*, where I
    is the mxm dimensional identity matrix, with

    :param u: m-dimensional numpy array
    :param v: m-dimensional numpy array
    """
    # Calculate uv^* (outer product)
    outer_uv = np.outer(u,v.conjugate())
    
    # Calculate v^*u (innter product)
    inner_uv = v.conjugate().dot(u)
    
    # Calculate A inverse
    Ainv = np.identity(len(u)) - (1 / (1 + inner_uv) ) * outer_uv
    
    # Return A inverse
    return Ainv


def timeable_rank1pert_inv():
    m = 600
    
    # Create variables u, v and A
    random.seed(1234*m)
    u = 1/np.sqrt(2)*(random.randn(m) + 1j*random.randn(m))
    v = 1/np.sqrt(2)*(random.randn(m) + 1j*random.randn(m))

    # Calculate uv^* (outer product)
    outer_uv = np.outer(u,v.conjugate())
    
    # Calculate v^*u (innter product)
    inner_uv = v.conjugate().dot(u)
    
    # Calculate A inverse
    Ainv = np.identity(len(u)) - (1 / (1 + inner_uv) ) * outer_uv  # noqa
  

def timeable_numpy_inverse():
    m = 600
    
    # Create variables u, v and A
    random.seed(1234*m)
    u = 1/np.sqrt(2)*(random.randn(m) + 1j*random.randn(m))
    v = 1/np.sqrt(2)*(random.randn(m) + 1j*random.randn(m))
    A = np.eye(m) + np.outer(u, v.conj())

    Ainv = np.linalg.inv(A) # noqa


def time_Ainv():
    """
    Get some timings for computing A inverse.
    """
    print("Timing for rank1pert_inv")
    print(timeit.Timer(timeable_rank1pert_inv).timeit(number=1))

    print("Timing for numpy inverse")
    print(timeit.Timer(timeable_numpy_inverse).timeit(number=1))
    
### time_Ainv()


def ABiC(Ahat, xr, xi):
    """Return the real and imaginary parts of z = A*x, where A = B + iC
    with

    :param Ahat: an mxm-dimensional numpy array with Ahat[i,j] = B[i,j] \
    for i<=j and Ahat[i,j] = C[i,j] for i>j.

    :return zr: m-dimensional numpy arrays containing the real part of z.
    :return zi: m-dimensional numpy arrays containing the imaginary part of z.
    """
    # construct matrix B from A hat
    upper_tri_a = np.triu(Ahat)
    B = upper_tri_a + upper_tri_a.T - np.diag(np.diag(upper_tri_a))
    
    # construct matrix C from A hat
    lower_tri_a = np.tril(Ahat, -1)
    C = lower_tri_a - lower_tri_a.T
    
    # initialise vector zr and zi to be vectors of length m
    zr = np.zeros(len(xr))
    zi = np.zeros(len(xi))
    
    # j cycles through 1 to m
    for j in range(1,len(xr)+1):
        
        # using slice notation to obtain the jth column of B and C where required
        zr += xr[j-1] * B[:,j-1] - xi[j-1] * C[:,j-1]
        zi += xr[j-1] * C[:,j-1] + xi[j-1] * B[:,j-1]
    
    # return the real and imaginary parts of z
    return zr, zi

