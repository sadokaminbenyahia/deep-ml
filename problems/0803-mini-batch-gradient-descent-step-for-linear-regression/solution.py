import numpy as np

def mini_batch_gd_step(X, y, weights, bias, batch_indices, lr):
    Xb = X[batch_indices]                 
    yb = y[batch_indices]                 
    m = len(batch_indices)

    error = Xb @ weights + bias - yb      

    grad_w = (2 / m) * (Xb.T @ error)     
    grad_b = (2 / m) * np.sum(error)      

    new_weights = weights - lr * grad_w
    new_bias = bias - lr * grad_b
    return np.append(new_weights, new_bias)