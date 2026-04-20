import numpy as np
import copy

def change_to_random_connections(net_arg, seed=None,fully_connected=False,probability=0.5):
    """
    This function creates random connections between the nodes.
    
    Args:
        net_arg (dict): The network dictionary
        seed (float): The seed for numpy random
        fully_connected (bool): Yes for all-connected network, False for random.
        probability (float): The probability to make the connection. If P(connection) > probability :=> We will have the connection.
    
    Returns:
        dict: The new dictionary with updated connections. 
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    net = copy.deepcopy(net_arg)
    names = net['names']
    R = net['R']
    R2 = np.zeros_like(R)
    made_connections = set()
    new_seq = ""
    
    for row in range(R.shape[0]):
        for col in range(R.shape[1]):
            if row != col:
                # Ensure each pair is considered only once (undirected logic, just for simplicity of simulation. We have an undirected graph.)
                if (col, row) not in made_connections and (row,col) not in made_connections:
                    if not fully_connected:
                        if np.random.random() > probability:
                            R2[row, col] = 1
                            made_connections.add((row, col))
                            new_seq += f"{names[row]}{names[col]}"
                    else:
                        R2[row, col] = 1
                        made_connections.add((row, col))
                        new_seq += f"{names[row]}{names[col]}"
    net['R'] = R2
    net['seq'] = new_seq
    
    return net
