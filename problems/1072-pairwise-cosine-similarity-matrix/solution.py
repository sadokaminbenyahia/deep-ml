import numpy as np

def pairwise_cosine_similarity(X):
    X = np.array(X, dtype=float)
    
    # Compute L2 norm of each row along axis 1
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    
    # Avoid division by zero by replacing zero norms with 1
    # (these rows will stay zero since their dot products are 0)
    safe_norms = np.where(norms == 0, 1.0, norms)
    
    # Normalize each vector to unit length
    X_normalized = X / safe_norms
    
    # Pairwise cosine similarity is the dot product of normalized vectors
    S = np.dot(X_normalized, X_normalized.T)
    
    # Zero out rows/columns that had a zero norm
    zero_mask = (norms.squeeze() == 0)
    S[zero_mask, :] = 0.0
    S[:, zero_mask] = 0.0
    
    # Round to 4 decimal places and convert to nested Python list
    return np.round(S, 4).tolist()