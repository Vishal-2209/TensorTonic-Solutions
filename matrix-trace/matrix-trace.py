import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    # np.trace or A.diagonal().sum() can be used
    x = 0
    for i in range(len(A)):
        x = x + A[i][i]
    return x