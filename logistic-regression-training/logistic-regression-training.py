import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    m, n = X.shape  # m = samples, n = features
    
    # 1. Initialize weights and bias with the correct dimensions
    w = np.zeros(n) 
    b = 0.0

    for i in range(steps):
        # 2. Forward pass: Linear combination then activation
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # 3. Compute gradients
        # error (p - y) is a vector of shape (m,)
        error = p - y
        
        # grad_w should be shape (n,), so we dot X.T (n, m) with error (m,)
        grad_w = np.dot(X.T, error) / m
        grad_b = np.sum(error) / m

        # 4. Update weights (Gradient Descent)
        w -= lr * grad_w
        b -= lr * grad_b
        
    return w, b