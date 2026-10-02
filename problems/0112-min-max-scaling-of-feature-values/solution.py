import numpy as np
def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    xv= np.array(x)
    min_v=xv.min(axis=0)
    max_v=xv.max(axis=0)
    scaled=(xv-min_v)/(max_v-min_v)
    return scaled.tolist()
    
    