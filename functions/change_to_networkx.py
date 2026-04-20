import networkx as nx

def networkX_graph(net):
    """
    Convert our network dictionary to networkx graph.
    
    Args:
        net (dict): Network dictionary
    
    Returns:
        object: NetwokrX graph.
    """
    
    positions = net['P']['states'][:, :2]  # Node positions
    nodes = net['init_data']['nodes']      # Number of nodes
    fixed_idx = net['fixed_idx']           # Indices of fixed nodes
    input_idx = net['input_idx']           # Indices of input nodes
    names = net['names']                   # Node labels
    R = net['R']                           # Adjacency matrix

    # Initialize a graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(range(nodes))

    # Get positions for nodes
    pos = {i: (positions[i, 0], positions[i, 1]) for i in range(nodes)}

    # Initialize a set to hold unique edges based on the adjacency matrix R
    edges = set()
    for i in range(nodes):
        for j in range(i + 1, nodes):  # Iterate only over upper triangle of the matrix
            if R[i, j] == 1:  # Add an edge if there's a connection
                edges.add((i, j))

    # Add edges to the graph
    G.add_edges_from(edges)

    return G