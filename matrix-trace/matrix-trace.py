import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    x = 0
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j] == A[j][i]:
                x = x + A[i][j]
    return x