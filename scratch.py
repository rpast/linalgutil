import linalgutil as lu
import numpy as np
import scipy.linalg as la

from sympy import Matrix, symbols, eye, zeros, simplify, factor, MatrixSymbol


A = np.array(
    [
        [1,2],
        [3,12],
        [4,5]
    ]
)