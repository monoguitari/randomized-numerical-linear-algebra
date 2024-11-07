import numpy as np

# https://numpy.org/doc/2.1/reference/random/generated/numpy.random.choice.html
# show pratyush the above, which does uniform sampling w/ or w/o replacement for us O.o
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
    n, d = A.shape
    r = np.max(48**2 * d * np.log(40*n*d) * np.log(100**2 * d * np.log(40*n*d)), (40 * d * np.log(40 * n * d)) / epsilon)

    # Creation of the samping-and-rescaling matrix S
    S = np.zeros((r, d))
    for t in range(r):
        i_t = np.random.choice(range(n))
        # for step 3, what is i? 
        e_i = np.zeros(n)
        e_i[i_t] = np.sqrt(n/r)
        S[t] = e_i @ A
    
    # normalized Random Hadamard Transform O(n log_2 r)
    H =  np.multiply(n, n)/ np.sqrt(n)
    D_diags = np.random.choice([-1, 1], size=n)
    D = np.diag(D_diags)

    # pinv = moore penrose pseudo inverse
    HDA = H @ D @ A
    HDb = H @ D @ b

    x_opt = np.linalg.pinv(S.T @ HDA) @ (S.T @ HDb)
    return x_opt
