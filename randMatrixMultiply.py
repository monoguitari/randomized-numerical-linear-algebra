import numpy as np
# from scipy.sparse import random as sparse_random
# from scipy.sparse import csr_matrix

def randMatrixMultiply(A: np.matrix, B: np.matrix, c: int, P_k: np.array) -> np.matrix:
    """ This function returns an estimator for the product AB using the random matrix multiplication algorithm

    :param A: The first m x n matrix to be multiplied
    :param B: The second n x p matrix to be multiplied
    :param c: The number of column-row pairs to choose
    :param P_k: The probability distribution to choose the probability matrix
    :return: The estimator for the product AB
    """
    n = A.shape[1]
    C = np.zeros((A.shape[0], c)) # m x c
    R = np.zeros((c, B.shape[1])) # c x p

    for t in range(c): # For t = 0 to c-1
        i_t = np.random.choice(range(n), p=P_k)
        coefficient = 1 / (np.sqrt(c * P_k[i_t]))
        C[:, t] = coefficient * A[:, i_t]
        R[t, :] = coefficient * B[i_t, :]
        
    return C @ R

def calculate_accuracy(A: np.matrix, B: np.matrix, c: int, P_k: np.array) -> float:
    """ This function calculates the accuracy of randMatrixMultiply compared to normal matrix multiplication

    :param A: The first m x n matrix to be multiplied
    :param B: The second n x p matrix to be multiplied
    :param c: The number of column-row pairs to choose
    :param P_k: The probability distribution to choose the probability matrix
    :return: The accuracy of the randMatrixMultiply function
    """
    # Compute the product using normal matrix multiplication
    AB_exact = A @ B
    
    # Compute the product using randMatrixMultiply
    AB_approx = randMatrixMultiply(A, B, c, P_k)
    
    # Calculate the Frobenius norm of the difference
    difference = AB_exact - AB_approx
    frobenius_norm = np.linalg.norm(difference, 'fro')
    
    # Calculate the Frobenius norm of the exact product
    frobenius_norm_exact = np.linalg.norm(AB_exact, 'fro')
    
    # Calculate the accuracy as 1 - (norm of difference / norm of exact product)
    accuracy = 1 - (frobenius_norm / frobenius_norm_exact)
    
    return accuracy

def calculate_loss(A: np.matrix, B: np.matrix, c: int, P_k: np.array) -> float:
    """ This function calculates the loss of randMatrixMultiply compared to normal matrix multiplication

    :param A: The first m x n matrix to be multiplied
    :param B: The second n x p matrix to be multiplied
    :param c: The number of column-row pairs to choose
    :param P_k: The probability distribution to choose the probability matrix
    :return: The loss of the randMatrixMultiply function
    """

    AB_exact = A @ B
    AB_approx = randMatrixMultiply(A, B, c, P_k)
    # Calculate the Frobenius norm of the difference
    difference = AB_exact - AB_approx
    frobenius_norm = np.linalg.norm(difference, 'fro')
    
    return frobenius_norm

def main():
    # Set the seed for reproducibility
    seed = 42
    np.random.seed(seed)
    
    # Set the dimensions of the matrices
    m = 1000
    n = 100
    p = 50
    
    # # Create the matrices A and B
    A = np.random.rand(m, n)
    B = np.random.rand(n, p)

    # Tall and sparse matrices
    # Create the sparse matrices A and B
    # density = 0.01  # 1% non-zero entries
    # A = sparse_random(m, n, density=density, format='csr', random_state=seed)
    # B = sparse_random(n, p, density=density, format='csr', random_state=seed)
    
    # Choose the number of column-row pairs to choose
    c = 2
    
    # Define the probability distribution P_k
    # P_k = np.full(n, 1/n)  # Uniform distribution for simplicity

    # Optimal Probability Distribution:
    A_col_norms = np.linalg.norm(A, axis=0)  
    B_row_norms = np.linalg.norm(B, axis=1)  
    P_k = (A_col_norms * B_row_norms) / np.sum(A_col_norms * B_row_norms)
    print(P_k)

    
    # Calculate the accuracy of the randMatrixMultiply function
    accuracy = calculate_accuracy(A, B, c, P_k)
    loss = calculate_loss(A, B, c, P_k)
    
    print(f"The accuracy of the randMatrixMultiply function is: {accuracy}")
    print(f"The loss of the randMatrixMultiply function is: {loss}")

if __name__ == "__main__":
    main()