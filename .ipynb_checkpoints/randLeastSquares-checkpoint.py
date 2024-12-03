import numpy as np
from scipy.linalg import hadamard



def randLeastSquares(A: np.matrix, b: np.array, epsilon: float):
    """
    Solves the least squares problem using a randLeastSquares algorithm.

    Parameters:
    A (np.matrix): The input matrix of shape (n, d).
    b (np.array): The target vector of shape (n).
    epsilon (float): The error tolerance for the approximation.

    Returns:
    np.array: The solution vector of shape (d).
    """
    # assumes A is full rank and n power of 2
    m, n = A.shape
    d = n
    
    # Ensure n is a power of 2
    # TODO: pad with 0s to get to the next power of 2
    if not (n > 0 and ((n & (n - 1)) == 0)):
        raise ValueError("n must be a positive integer, and n must be a power of 2")
    
    r = max(48**2 * d * np.log(40*n*d) * np.log(100**2 * d * np.log(40*n*d)), (40 * d * np.log(40 * n * d)) / epsilon)
    print("r", r)
    r = int(np.ceil(r))

    # Creation of the samping-and-rescaling matrix S

    # Bottleneck in creating the sparse matrix, takes up the majority of the algorithm's time
    S = np.zeros((r, d))
    for t in range(r):
        i_t = np.random.choice(range(n))
        # for step 3, what is i? 
        e_i = np.zeros(n)
        e_i[i_t] = np.sqrt(n/r)
        S[t] = e_i
    H = hadamard(n) / np.sqrt(n)
    D_diags = np.random.choice([-1, 1], size=n)
    D = np.diag(D_diags)

    # Compute HDA and HDb
    HDA = H @ D @ A
    HDb = H @ D @ b

    # Generate a random sampling matrix S
    # S = np.random.randn(r, m)

    # Compute the solution vector x_opt
    x_opt = np.linalg.pinv(S @ HDA) @ (S @ HDb)
    return x_opt

from tqdm import tqdm

def relative_err(x_opt, x_true):
    return np.linalg.norm(x_opt - x_true) / np.linalg.norm(x_true)

# Parameters
num_tests = 1
matrix_size = (4, 4)
epsilon = 0.99

# Run multiple tests
err = []
for _ in tqdm(range(num_tests)):
    A = np.random.randn(*matrix_size)
    b = np.random.randn(matrix_size[0])
    
    x_opt = randLeastSquares(A, b, epsilon)
    x_true = np.linalg.lstsq(A, b, rcond=None)[0]
    
    acc = relative_err(x_opt, x_true)
    err.append(acc)

# Compute average accuracy
average_err = np.mean(err)
print("Average error over {} tests: {}".format(num_tests, average_err))