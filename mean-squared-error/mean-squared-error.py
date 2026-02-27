import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    a = 0
    for i in range(len(y_pred)):
        y = y_pred[i] - y_true[i]
        t = y ** 2
        a = a + t
    return a / len(y_pred)
