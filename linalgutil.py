"""
A set of utilities to learn linear algebra concepts.

This module is a companion to the "Linear Algebra and Learning from Data" book
by Gilbert Strang, providing functions to visualize and understand linear algebra concepts.
"""

import numpy as np
from sympy import Matrix

if __name__ == "__main__":
    print("This module is not meant to be run directly. Import it to use its functions.")


## Functions

def decompose_matrix(A):
    """Decompose matrix A into its column space, R, and null space.
    :param A: Input matrix (numpy array or sympy Matrix)
    :return: Tuple of matrices (C, R, N) where:
        - C is the column space matrix (m×r, full column rank)
        - R is the matrix such that CR = A (r×n)
        - N is the null space matrix (n×(n−r), full row rank)
    :rtype: tuple of numpy matrices
    :raises ValueError: If A is not a 2D matrix.
    :raises TypeError: If A is not a numpy array or sympy Matrix.
    """

    if not isinstance(A, (np.ndarray, Matrix)):
        raise TypeError("Input A must be a numpy array or sympy Matrix.")
    if A.ndim != 2:
        raise ValueError("Input A must be a 2D matrix.")
    
    A_sym = Matrix(A)
    
    # Get the column space
    col_space = A_sym.columnspace()
    
    # Get the null space
    null_space = A_sym.nullspace()


    col_space_matrix = Matrix.hstack(*col_space)  # C is m×r, full column rank
    null_space_matrix = Matrix.hstack(*null_space)  # N is n×(n−r), full row rank

    # Compute the R matrix such that CR = A
    r_echelon = col_space_matrix.pinv() * A_sym

    # Convert to numpy matrices
    colspace_np = np.asarray(col_space_matrix, dtype=float)
    rrefa_np = np.asarray(r_echelon, dtype=float)
    nspace_np = np.asarray(null_space_matrix, dtype=float)

    np.testing.assert_allclose(colspace_np @ rrefa_np, A, rtol=1e-8, atol=1e-10) 
    
    return colspace_np, rrefa_np, nspace_np


def ax0_solutions(A):
    """
    Find how many unique solutions exist for homogenous system Ax = 0.
    """

    sols = A.shape[1] - np.linalg.matrix_rank(A)

    if sols == 0:
        return "There is only the trivial solution x = 0."
    elif sols > 0:
        return f"There are {sols} unique solutions for the homogenous system Ax = 0."
    
    return sols
    

def learn_subspace_dims(A):
    """
    Learn the dimensions of the column space and null space of matrix A.
    
    :param A: Input matrix (numpy array or sympy Matrix)
    :return: Tuple of dimensions (dim_col_space, dim_null_space)
    :rtype: tuple
    """
    
    if not isinstance(A, (np.ndarray, Matrix)):
        raise TypeError("Input A must be a numpy array or sympy Matrix.")
    
    colspace_dim = np.linalg.matrix_rank(A)
    rowspace_dim = np.linalg.matrix_rank(A.T)
    nullspace_dim = A.shape[1] - colspace_dim
    nullspace_t_dim = A.shape[0] - rowspace_dim

    print(f"Column space dimension: {colspace_dim}")
    print(f"Row space dimension: {rowspace_dim}")
    print(f"Null space dimension: {nullspace_dim}")
    print(f"Null space transpose dimension: {nullspace_t_dim}")

    return colspace_dim, nullspace_dim, rowspace_dim, nullspace_t_dim