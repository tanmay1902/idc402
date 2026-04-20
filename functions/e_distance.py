import numpy as np
import numpy.random as random

def e_distance(p1, p2):
    """
    Calculate eucledian distance between two points.
    
    Args:
        p1 (list): The [x,y] list of coordinates of point 1.
        p2 (list): The [x,y] list of coordinates of point 2.
        
    Returns:
        float: The scalar-distance between both points.
        float: Normalized distance between both points.
        
    """
    
    dx = (p2[0] - p1[0])
    dy = (p2[1] - p1[1])
    #print(dx,dy)
    d = np.sqrt(dx**2 + dy**2)

    if d == 0:
       norm_dir = np.zeros_like(p1)
    else:
        norm_dir = (p2 - p1) / d

    return d, norm_dir

