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
    colspace_np = np.matrix(col_space_matrix)
    rrefa_np = np.matrix(r_echelon)
    nspace_np = np.matrix(null_space_matrix)

    assert colspace_np @ rrefa_np == A, "CR does not equal A"
    
    return colspace_np, rrefa_np, nspace_np

