import numpy as np
import numpy.random as random

def rand_in_range(range_val, a=None):
    """
    Generates a random number within the specified range.
    
    Args: 
        range_val (list): The list of [lower,upper] limit of range
        a (int): Size of the numpy array of random values.
    
    Returns:
        np.array: The numpy array of the random values.
    """
    lo, up = range_val
    if a is None:
        return np.random.uniform(lo, up)
    else:
        return np.random.uniform(lo, up, size=a)