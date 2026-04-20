import numpy as np
from numpy import random as random
from itertools import product
from functions.e_distance import e_distance
from functions.random import rand_in_range
from copy import deepcopy

def init_ms_system(parameters):
	
    """
    Initialize the configuration dictionary for a mass-spring reservoir system.

    This function sets up initial conditions and metadata required to build and simulate
    a 2D particle-based network (DNA spring network). It includes spatial layout, 
    input/output connectivity, time step, and physical constants.

    Args:
        parameters (dict): Dictionary of simulation parameters, must include:
            - 'input_nodes' (int): Number of input nodes.
            - 'readout' (str): Readout method ('POSITIONS' or 'LENGTHS').
            - 'A' (float): Persistence length or spring constant factor.
            - 'L' (list): Contour length range [L_min, L_max].
            - 'nodes' (int): Total number of particles (nodes) in the system.
            - 'x_lim' (list): x-axis spatial bounds [xmin, xmax].
            - 'y_lim' (list): y-axis spatial bounds [ymin, ymax].
            - 'rk_steps' (int): Runge-Kutta steps per integration.
            - 'basepair_range' (list): Range of base pairs per spring.
            - 'time_step' (float): Simulation time step in seconds.
            - 'show_steps' (int): Interval for logging progress.
            - 'input_length' (int): Length of the input sequence.
            - 'b' (float): Bead radius or related physical parameter.

    Returns:
        dict: Dictionary `init_dict` containing all initialized fields for building the system:
            - 'nodes': Number of masses/nodes
            - 'positions': Random initial 2D positions and zero velocities
            - 'x_lim', 'y_lim': Spatial limits
            - 'fixed_positions': Whether any nodes are fixed (False by default)
            - 'nInput', 'input_percentage': Number and fraction of input nodes
            - 'nOutput', 'out_conn': Output setup (currently 1 output connected to all nodes)
            - 'w_out_range', 'w_feedback_range', 'w_input_range': Weight initialization ranges
            - 'time_step', 'show_steps': Simulation timing configuration
            - 'readout_type': Determines if the output is based on 'POSITIONS' or 'LENGTHS'
            - 'persistence': Spring persistence parameter (A)
            - 'contour_range': Spring contour length range
            - 'rk_steps': Number of Runge-Kutta steps
            - 'basepair_range': Base pair range for DNA spring modeling
            - 'parameters': Nested dictionary for additional constants (e.g., 'b')
    """
    inputs = parameters['input_nodes']
    readout = parameters['readout']
    A = parameters['A']
    L = parameters['L']
    nodes = parameters['nodes']
    x_lim = parameters['x_lim']
    y_lim = parameters['y_lim']
    rk_steps = parameters['rk_steps']
    basepair_range = parameters['basepair_range']
    
    init_dict = dict()
    init_dict['nodes'] = nodes #Number of masses
    
    percentage = inputs/nodes #percentage to set for inputs
    
    init_dict['x_lim'] = x_lim
    init_dict['y_lim'] = y_lim
    
    init_dict['positions'] = np.zeros((init_dict['nodes'],4))
    init_dict['positions'][:,0] = np.random.uniform(init_dict['x_lim'][0], init_dict['x_lim'][1], init_dict['nodes'])
    init_dict['positions'][:,1] = np.random.uniform(init_dict['y_lim'][0], init_dict['y_lim'][1], init_dict['nodes'])
    
    init_dict['fixed_positions'] = False
    
    init_dict['w_out_range'] = [1, 1]
    init_dict['w_feedback_range'] = [-1, 1]
    init_dict['w_input_range'] = [-1 ,1]
    
    init_dict['fb_conn'] = 0 #Feedback connectivity
    
    init_dict['nInput'] = inputs
    init_dict['input_percentage'] = percentage #percentage of input connectivity nodes
    init_dict['nOutput'] = 1
    init_dict['out_conn'] = 1 #1 = 100% of nodes
    
    init_dict['time_step'] = parameters['time_step'] # time_step = 1 ms = 0.001 s
    init_dict['show_steps'] = parameters['show_steps'] # to show the simulation progress at every 1000 steps
    
    init_dict['readout_type'] = readout
    init_dict['persistence'] = A
    init_dict['contour_range'] = L
    init_dict['input_length'] =  parameters['input_length']
    init_dict['rk_steps'] = rk_steps
    init_dict['basepair_range'] = basepair_range
    init_dict['parameters'] = dict()
    init_dict['parameters']['b'] = parameters['b']
    
    return init_dict

def get_random_names(num):
    """
    Generate `num` unique DNA-like sequence names using combinations of 'A', 'T', 'G', 'C'.

    The function dynamically determines the minimal sequence length (`num_characters`)
    required to generate at least `num` unique names using nucleotide characters.
    It ensures that the total number of possible combinations (4^N) is sufficient.

    Args:
        num (int): Number of unique sequence names required.

    Returns:
        tuple:
            - combinations (list of str): List of `num` DNA-like sequence names.
            - num_characters (int): Length of each sequence used to generate the names.

    Example:
        >>> get_random_names(10)
        (['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 'GA', 'GT'], 2)

    Notes:
        - If `num == 4`, returns 1-letter sequences: ['A', 'T', 'G', 'C']
        - Uses itertools.product to enumerate all combinations
    """
    
    letters = [l**4 for l in range(2,10)]
    letters += [4]
    
    #determine number of letters required in sequence
    if num in letters:
        if num != 4:
            num_characters = num
        else:
            num_characters = 1
    else:
        #now we sort the sequence det list with number of nodes, and get just higher one, to determine how much letters we need.
        sequence_det_list = np.sort(letters + [num])
        index0fNode = np.where(sequence_det_list == num)[0][0]
        powered  = sequence_det_list[index0fNode+1]
        if powered != 4:
            num_characters = int(powered**(0.25))
        else:
            num_characters = 1
    
    characters = "ATGC"
    combinations = [''.join(pair) for pair in product(characters, repeat=num_characters)]  
    
    return combinations[:num],num_characters

def network_system(init_func,parameters,names_pre_def=None):
    
    """
    Construct a spring-based DNA-inspired network system for physical reservoir computing.

    This function initializes a network of particles (nodes) connected by nonlinear springs.
    The connectivity is determined using synthetic DNA-like sequences. It also sets up 
    input, feedback, and output weights, and assigns initial positions and velocities.

    Args:
        init_func (function): A function that initializes simulation parameters and 
                              returns an `init_dict` dictionary.
        parameters (dict): Dictionary containing system-level parameters like node count,
                           spatial limits, spring constants, input/output ranges, DNA sequence, etc.
        names_pre_def (list, optional): List of predefined node names. If not provided,
                                        names are generated automatically using base pairs.

    Returns:
        dict: A dictionary `net` containing the complete state of the network, including:
            - `P`: Node states (position, velocity, fixed status)
            - `W`: Spring and connectivity weights
            - `W_in`, `W_out`, `W_fb`: Input, output, and feedback connection matrices
            - `R`: Raw connectivity matrix
            - `names`: DNA-like names for each node
            - `input`: Dictionary with the name and index of the input node
            - `fixed_idx`, `input_idx`, `output_idx`: Indices of fixed, input, and output nodes
            - `readout_type`: Type of readout used ('POSITIONS' or 'LENGTHS')
            - `rk_steps`: Runge-Kutta steps
            - `init_data`: Original initialization dictionary

    Notes:
        - The connections are derived from a provided DNA sequence string.
        - Ensures at least one input node is not fixed.
        - Each spring has parameters like persistence length, contour length, and rest length.
        - This network is designed to be used with downstream simulation functions.

    Example:
        >>> net = network_system(init_ms_system, parameters)
    """
    
    net = dict()
    net['init_data'] = init_dict = init_func(parameters)
    num = nodes = init_dict['nodes']
    
    net['P'] = dict()
    
    if init_dict['fixed_positions']:
        net['P']['states'] = init_dict['positions']
    else:
        net['P']['states'] = np.column_stack([
            random.uniform(init_dict['x_lim'][0], init_dict['x_lim'][1], num),
			random.uniform(init_dict['y_lim'][0], init_dict['y_lim'][1], num),
			np.zeros(num),
			np.zeros(num)
        ])
    
    net['P']['force'] = np.zeros((num,2))
    net['P']['fixed'] = np.zeros(num)
    
    net['init_data']['P'] = net['P']
    net['pos_noise'] = 0
    net['dist_noise'] = 0
    
    min_idx = np.argmin(net['P']['states'][:, 0])
    max_idx = np.argmax(net['P']['states'][:, 0])

    if nodes < 3:
        net['P']['fixed'][min_idx] = 1
    else:
        net['P']['fixed'][min_idx] = 1
        net['P']['fixed'][max_idx] = 1
    
    #get random sequence for nodes, generalized.
    names,num_char = get_random_names(num)
    if names_pre_def != None:
        net['names'] = names_pre_def
    else:
        net['names'] = names
    net['num_char'] = num_char
    input_name = np.random.choice(names)
    index_of_input_name = names.index(input_name)
    if net['P']['fixed'][index_of_input_name] == 1:
        for ijk in range(len(net['P']['fixed'])):
            if net['P']['fixed'][ijk] == 0 and names[ijk] != input_name:
                names[ijk], names[index_of_input_name] = names[index_of_input_name], names[ijk]
                index_of_input_name = ijk
                break
    net['input'] = dict([('name',""), ('index','')])
    net['input']['name'] = input_name
    net['input']['index'] = index_of_input_name
    #get connection sequence from parameter
    connection_string = parameters['sequence']
                    
    
    connection_matrix = np.zeros((nodes,nodes))
    #get fragments of size num_char * 2 to have 2 nodes in that fragment    
    fragments = []
    for i in range(0,len(connection_string),num_char*2):
        fragments.append(connection_string[i:i+num_char*2])
    
    #matrix to see if the connections are already present or not
    established_connections = []
    
    #making connections
    for first_node in range(len(names)):
        for second_node in range(len(names)):
            if first_node != second_node:
                possible_fragment = names[first_node]+names[second_node]
                if possible_fragment in fragments:
                    if (first_node,second_node) not in established_connections or (second_node,first_node) not in established_connections:
                        established_connections.append((first_node,second_node))
                        connection_matrix[first_node,second_node] = 1
    net['R'] = R = connection_matrix
    
    total_connections = 0
    for row in range(len(connection_matrix)):
        for col in range(len(connection_matrix[row])):
            if connection_matrix[row][col] == 1:
                total_connections += 2
                
    w_num = int(np.sum(R))
    net['W'] = dict()
    net['W']['avgcon'] = total_connections/8
    net['W']['from'] = np.zeros(w_num)
    net['W']['to'] = np.zeros(w_num)
    
    net['W']['k1'] = net['W']['k3'] = net['W']['d1'] = net['W']['d3']  = rand_in_range([0,1], w_num)  
    
    #Initializing Spring Parameters
    net['W']['bp'] = rand_in_range(init_dict['basepair_range'],w_num)
    net['W']['A'] = init_dict['persistence']
    net['W']['L'] = net['W']['bp']*0.63e-9
    net['W']['E'] = np.zeros(w_num)
    net['W']['l0'] = np.zeros(w_num) #The initial length of springs, same size as number of connections
    net['W']['dist_old'] = np.zeros(w_num) #The old distance, i.e. distance at t-delta(t) seconds.same size as number of connections
    
    w_idx  = 0
    
    for i in range(init_dict['nodes']):
        for j in range(init_dict['nodes']):
            if R[i, j] == 1:
                net['W']['from'][w_idx] = i
                net['W']['to'][w_idx] = j
                p1 =net['P']['states'][i, :2]
                p2 =net['P']['states'][j,:2]
                net['W']['l0'][w_idx] = e_distance(p1,p2)[0]
                w_idx += 1
 
    net['W']['E'][:] = net['W']['dist_old'][:] = net['W']['l0'][:]
    net['W_in'] = np.zeros((init_dict['nodes'], init_dict['nInput']))
    nmax = int(init_dict['nodes'] * init_dict['input_percentage'])
     
    net['W_in'][index_of_input_name] = random.uniform(init_dict['w_input_range'][0], init_dict['w_input_range'][1])
    
    
    Wins = []
    fixeds = []
    for win in net['W_in']:
        Wins.append(float(win))
    
    
    for pxs in net['P']['fixed']:
        fixeds.append(int(pxs))

    Wins = np.array(Wins)
    fixeds = np.array(fixeds)
    
    if not (Wins.any() > 0.):
        argmin = np.argmin(fixeds)
        net['W_in'][argmin] = random.uniform(init_dict['w_input_range'][0], init_dict['w_input_range'][1], 1)
    
    net['W_fb'] = np.zeros((init_dict['nodes'], init_dict['nOutput']))
    
    nmax = int(np.ceil(init_dict['nodes'] * init_dict['fb_conn']))
    
    for nI in range(init_dict['nOutput']):
        Idx = random.permutation(init_dict['nodes'])
        Idx = Idx[:nmax]
        net['W_fb'][Idx,nI] = random.uniform(init_dict['w_feedback_range'][0], init_dict['w_feedback_range'][1], nmax)
    
    if init_dict['readout_type'] == 'POSITIONS':
        net['W_out'] = np.zeros((init_dict['nodes'], init_dict['nOutput']))
        nmax = int(init_dict['nodes']*init_dict['out_conn'])
        
        for nO in range(0,init_dict['nOutput']):
            Odx = random.permutation(init_dict['nodes']) #random input connections
            Odx[nmax+1:] = []
            net['W_out'][Odx,nO] = rand_in_range(init_dict['w_out_range'], nmax) #between -iScale and +iScale
    
    else:
        if init_dict['readout_type'] == 'LENGTHS':
            w_num = (net['W']['l0']).shape[0]
            net['W_out'] = np.zeros((w_num, init_dict['nOutput']))
            nmax = int(np.ceil(w_num * init_dict['out_conn']))
            
            for nO in range(init_dict['nOutput']):
                Odx = random.permutation(w_num)
                Odx[nmax+1:] = []
                net['W_out'][Odx, nO] = rand_in_range(init_dict['w_out_range'], a=nmax)
        
        else:
            print('ERROR - unknown output chosen')
    
    net['readout_type'] = init_dict['readout_type']
    net['fixed_idx'] = np.where(net['P']['fixed'] == 1)[0]
    #print(net['fixed_idx'])
    net['input_idx'] = np.where(np.sum(net['W_in'], axis=1) != 0)[0]
    net['output_idx'] = np.where(np.sum(net['W_out'], axis=1) != 0)[0]
    
    net['rk_steps'] = net['init_data']['rk_steps']
    
    net['W']['nConnection'] = net['W']['k1'].shape[0]
	
    
    return net


def network_change(net_prev,sequence):
    
    """
    This function changes the connections according to the sequence.
    
    Args:
        net_prev (dict): Network to be changed
        sequence (str): Coding strand
        
    Returns:
        dict: New dictionary of network
    """
    if isinstance(sequence,list):
        sequence = sequence[0]
    
    net = dict()
    net['init_data'] = init_dict = net_prev['init_data']
    num = nodes = init_dict['nodes']
    
    net['P'] = net_prev['P']
    
    net['P']['force'] = np.zeros((num,2))
    net['P']['fixed'] = np.zeros(num)
    
    net['init_data']['P'] = net['P']
    net['pos_noise'] = 0
    net['dist_noise'] = 0
    
    min_idx = np.argmin(net['P']['states'][:, 0])
    max_idx = np.argmax(net['P']['states'][:, 0])

    if nodes < 3:
        net['P']['fixed'][min_idx] = 1
    else:
        net['P']['fixed'][min_idx] = 1
        net['P']['fixed'][max_idx] = 1
    
    #get random sequence for nodes, generalized.
    _,num_char = get_random_names(num)
    names = net_prev['names']
    net['names'] = names
    net['num_char'] = num_char
    net['input'] = dict([('name',""), ('index','')])
    index_of_input_name = net['input']['index'] = net_prev['input']['index']
    
    #get connection sequence from parameter
    connection_string = sequence
    
    connection_matrix = np.zeros((nodes,nodes))
    #get fragments of size num_char * 2 to have 2 nodes in that fragment    
    fragments = []
    for i in range(0,len(connection_string),num_char*2):
        fragments.append(connection_string[i:i+num_char*2])
    
    #matrix to see if the connections are already present or not
    established_connections = []
    
    for first_node in range(len(names)):
        for second_node in range(len(names)):
            if first_node != second_node:
                possible_fragment = names[first_node]+names[second_node]
                if possible_fragment in fragments:
                    if (first_node,second_node) not in established_connections or (second_node,first_node) not in established_connections:
                        established_connections.append((first_node,second_node))
                        connection_matrix[first_node,second_node] = 1
    
    net['R'] = R = connection_matrix
    w_num = int(np.sum(R))
    net['W'] = dict()
    net['W']['avgcon'] = np.sum(R)/8
    net['W']['from'] = np.zeros(w_num)
    net['W']['to'] = np.zeros(w_num)
    
    net['W']['k1'] = net['W']['k3'] = net['W']['d1'] = net['W']['d3']  = rand_in_range([0,1], w_num)  
    
    """
    
    Now we define the spring parameters.
    
        'bp': base-pairs of springs
        'A':  persistence lengths
        'L':  contour lengths, defined as 
                Lc = 0.63 * bp nm
                by default we use bp, and assume it is of order 10^3. so we directly use Lc=0.63*bp micron
        'E': Spring lengths at t=0
        'l0':Spring lengths at t=0
    
    """
    net['W']['bp'] = rand_in_range(init_dict['basepair_range'],w_num)
    net['W']['A'] = init_dict['persistence']
    net['W']['L'] = net['W']['bp']*0.63e-9
    net['W']['E'] = np.zeros(w_num)
    net['W']['l0'] = np.zeros(w_num)
    
    net['W']['dist_old'] = np.zeros(w_num)
    
    w_idx  = 0
    
    for i in range(init_dict['nodes']):
        for j in range(init_dict['nodes']):
            if R[i, j] == 1:
                net['W']['from'][w_idx] = i
                net['W']['to'][w_idx] = j
                p1 =net['P']['states'][i, :2]
                p2 =net['P']['states'][j,:2]
                net['W']['l0'][w_idx] = e_distance(p1,p2)[0]
                w_idx += 1
 
    net['W']['E'][:] = net['W']['dist_old'][:] = net['W']['l0'][:]
    net['W_in'] = np.zeros((init_dict['nodes'], init_dict['nInput']))
    nmax = int(init_dict['nodes'] * init_dict['input_percentage'])
    
    net['W_in'][index_of_input_name] = random.uniform(init_dict['w_input_range'][0], init_dict['w_input_range'][1])
    
    
    Wins = []
    fixeds = []
    for win in net['W_in']:
        Wins.append(float(win))
    
    for pxs in net['P']['fixed']:
        fixeds.append(int(pxs))

    Wins = np.array(Wins)
    fixeds = np.array(fixeds)
    
    net['W_in'] = net_prev['W_in']

    net['W_fb'] = np.zeros((init_dict['nodes'], init_dict['nOutput']))
    
    nmax = int(np.ceil(init_dict['nodes'] * init_dict['fb_conn']))
    
    for nI in range(init_dict['nOutput']):
        Idx = random.permutation(init_dict['nodes'])
        Idx = Idx[:nmax]
        net['W_fb'][Idx,nI] = random.uniform(init_dict['w_feedback_range'][0], init_dict['w_feedback_range'][1], nmax)
    
    if init_dict['readout_type'] == 'POSITIONS':
        net['W_out'] = np.zeros((init_dict['nodes'], init_dict['nOutput']))
        nmax = int(init_dict['nodes']*init_dict['out_conn'])
        
        for nO in range(0,init_dict['nOutput']):
            Odx = random.permutation(init_dict['nodes']) #random input connections
            Odx[nmax+1:] = []
            net['W_out'][Odx,nO] = rand_in_range(init_dict['w_out_range'], nmax) #between -iScale and +iScale
    
    else:
        if init_dict['readout_type'] == 'LENGTHS':
            w_num = (net['W']['l0']).shape[0]
            net['W_out'] = np.zeros((w_num, init_dict['nOutput']))
            nmax = int(np.ceil(w_num * init_dict['out_conn']))
            
            for nO in range(init_dict['nOutput']):
                Odx = random.permutation(w_num)
                Odx[nmax+1:] = []
                net['W_out'][Odx, nO] = rand_in_range(init_dict['w_out_range'], a=nmax)
        
        else:
            print('ERROR - unknown output chosen')
    
    #Store the readout type
    net['readout_type'] = init_dict['readout_type']
    
    #Store the fixed node index
    net['fixed_idx'] = np.where(net['P']['fixed'] == 1)[0]
    
    #Store the input node index
    net['input_idx'] = np.where(np.sum(net['W_in'], axis=1) != 0)[0]
    
    #Store the output node index
    net['output_idx'] = np.where(np.sum(net['W_out'], axis=1) != 0)[0]
    
    #Store the runge kutta step
    net['rk_steps'] = net['init_data']['rk_steps']
    
    #Store number of connections
    net['W']['nConnection'] = net['W']['k1'].shape[0]
	
 
    return net


# -------------------------------------------------------------
#  NETWORK SYSTEM WITH DELAUNAY TRIANGULATION
# -------------------------------------------------------------
def network_system_delaunay(init_func, parameters):

    net = dict()
    net['init_data'] = init_dict = init_func(parameters)
    num = nodes = init_dict['nodes']

    # -------------------------------------------------------------
    # POSITIONS
    # -------------------------------------------------------------
    net['P'] = dict()

    if init_dict['fixed_positions']:
        net['P']['states'] = init_dict['positions']
    else:
        net['P']['states'] = np.column_stack([
            random.uniform(init_dict['x_lim'][0], init_dict['x_lim'][1], num),
            random.uniform(init_dict['y_lim'][0], init_dict['y_lim'][1], num),
            np.zeros(num),
            np.zeros(num)
        ])

    net['P']['force'] = np.zeros((num, 2))
    net['P']['fixed'] = np.zeros(num)

    net['init_data']['P'] = net['P']
    net['pos_noise'] = 0
    net['dist_noise'] = 0

    # Fix two extreme nodes
    min_idx = np.argmin(net['P']['states'][:, 0])
    max_idx = np.argmax(net['P']['states'][:, 0])
    net['P']['fixed'][min_idx] = 1
    if nodes >= 3:
        net['P']['fixed'][max_idx] = 1

    # -------------------------------------------------------------
    # NAMES & INPUT MAPPING
    # -------------------------------------------------------------
    names, num_char = get_random_names(num)
    net['names'] = names
    net['num_char'] = num_char

    input_name = parameters['sequence'][:num_char]

    if input_name not in names:
        for name_index in range(len(names)):
            if net['P']['fixed'][name_index] == 0:
                names[name_index] = input_name
                index_of_input_name = name_index
                break
    else:
        index_of_input_name = names.index(input_name)
        if net['P']['fixed'][index_of_input_name] == 1:
            for ijk in range(len(net['P']['fixed'])):
                if net['P']['fixed'][ijk] == 0 and names[ijk] != input_name:
                    names[ijk], names[index_of_input_name] = names[index_of_input_name], names[ijk]
                    index_of_input_name = ijk
                    break

    net['input'] = {'name': input_name, 'index': index_of_input_name}

    # -------------------------------------------------------------
    # DELAUNAY TRIANGULATION CONNECTIONS
    # -------------------------------------------------------------
    pts = net['P']['states'][:, :2]
    tri = Delaunay(pts)
    connection_matrix = np.zeros((nodes, nodes))

    for simplex in tri.simplices:
        i, j, k = simplex
        connection_matrix[i, j] = connection_matrix[j, i] = 1
        connection_matrix[i, k] = connection_matrix[k, i] = 1
        connection_matrix[j, k] = connection_matrix[k, j] = 1

    net['R'] = R = connection_matrix
    print(R)
    # -------------------------------------------------------------
    # SPRING CONNECTION LIST
    # -------------------------------------------------------------
    w_num = int(np.sum(R))
    net['W'] = dict()
    net['W']['from'] = np.zeros(w_num)
    net['W']['to'] = np.zeros(w_num)

    net['W']['k1'] = net['W']['k3'] = net['W']['d1'] = net['W']['d3'] = rand_in_range([0, 1], w_num)

    # Spring parameters
    net['W']['bp'] = rand_in_range(init_dict['basepair_range'], w_num)
    net['W']['A'] = init_dict['persistence']
    net['W']['L'] = net['W']['bp'] * 0.63e-9
    net['W']['E'] = np.zeros(w_num)
    net['W']['l0'] = np.zeros(w_num)
    net['W']['dist_old'] = np.zeros(w_num)

    w_idx = 0
    for i in range(nodes):
        for j in range(nodes):
            if R[i, j] == 1:
                net['W']['from'][w_idx] = i
                net['W']['to'][w_idx] = j
                p1 = net['P']['states'][i, :2]
                p2 = net['P']['states'][j, :2]
                net['W']['l0'][w_idx] = e_distance(p1, p2)[0]
                w_idx += 1

    net['W']['E'][:] = net['W']['dist_old'][:] = net['W']['l0'][:]
    net['W']['nConnection'] = w_num

    # -------------------------------------------------------------
    # INPUT WEIGHTS
    # -------------------------------------------------------------
    net['W_in'] = np.zeros((nodes, init_dict['nInput']))
    net['W_in'][index_of_input_name] = random.uniform(
        init_dict['w_input_range'][0], init_dict['w_input_range'][1]
    )

    # -------------------------------------------------------------
    # FEEDBACK WEIGHTS
    # -------------------------------------------------------------
    net['W_fb'] = np.zeros((nodes, init_dict['nOutput']))
    nmax = int(np.ceil(nodes * init_dict['fb_conn']))

    for nI in range(init_dict['nOutput']):
        Idx = random.permutation(nodes)[:nmax]
        net['W_fb'][Idx, nI] = random.uniform(
            init_dict['w_feedback_range'][0],
            init_dict['w_feedback_range'][1],
            nmax
        )

    # -------------------------------------------------------------
    # READOUT WEIGHTS
    # -------------------------------------------------------------
    if init_dict['readout_type'] == 'POSITIONS':
        net['W_out'] = np.zeros((nodes, init_dict['nOutput']))
        nmax = int(nodes * init_dict['out_conn'])

        for nO in range(init_dict['nOutput']):
            Odx = random.permutation(nodes)[:nmax]
            net['W_out'][Odx, nO] = rand_in_range(init_dict['w_out_range'], nmax)

    elif init_dict['readout_type'] == 'LENGTHS':
        w_num = net['W']['l0'].shape[0]
        net['W_out'] = np.zeros((w_num, init_dict['nOutput']))
        nmax = int(np.ceil(w_num * init_dict['out_conn']))

        for nO in range(init_dict['nOutput']):
            Odx = random.permutation(w_num)[:nmax]
            net['W_out'][Odx, nO] = rand_in_range(init_dict['w_out_range'], a=nmax)

    else:
        print("ERROR - unknown output type")

    # Indices
    net['readout_type'] = init_dict['readout_type']
    net['fixed_idx'] = np.where(net['P']['fixed'] == 1)[0]
    net['input_idx'] = np.where(np.sum(net['W_in'], axis=1) != 0)[0]
    net['output_idx'] = np.where(np.sum(net['W_out'], axis=1) != 0)[0]

    net['rk_steps'] = init_dict['rk_steps']

    return net



# -------------------------------------------------------------
#  CHANGE NETWORK WITH DELAUNAY TRIANGULATION
# -------------------------------------------------------------
def network_change_delaunay(net_prev, sequence):

    if isinstance(sequence, list):
        sequence = sequence[0]

    net = dict()
    net['init_data'] = init_dict = net_prev['init_data']
    num = nodes = init_dict['nodes']

    net['P'] = net_prev['P']

    net['P']['force'] = np.zeros((num, 2))
    net['P']['fixed'] = np.zeros(num)

    net['init_data']['P'] = net['P']
    net['pos_noise'] = 0
    net['dist_noise'] = 0

    # Fix nodes
    min_idx = np.argmin(net['P']['states'][:, 0])
    max_idx = np.argmax(net['P']['states'][:, 0])
    net['P']['fixed'][min_idx] = 1
    if nodes >= 3:
        net['P']['fixed'][max_idx] = 1

    # Names & input
    names = net_prev['names']
    net['names'] = names
    net['num_char'] = net_prev['num_char']
    index_of_input_name = net_prev['input']['index']
    net['input'] = net_prev['input']

    # -------------------------------------------------------------
    # DELAUNAY CONNECTION MATRIX
    # -------------------------------------------------------------
    pts = net['P']['states'][:, :2]
    tri = Delaunay(pts)
    connection_matrix = np.zeros((nodes, nodes))

    for simplex in tri.simplices:
        i, j, k = simplex
        connection_matrix[i, j] = connection_matrix[j, i] = 1
        connection_matrix[i, k] = connection_matrix[k, i] = 1
        connection_matrix[j, k] = connection_matrix[k, j] = 1

    net['R'] = R = connection_matrix
    print(R)
    # -------------------------------------------------------------
    # SPRING PARAMETERS
    # -------------------------------------------------------------
    w_num = int(np.sum(R))
    net['W'] = dict()
    net['W']['avgcon'] = np.sum(R) / 8
    net['W']['from'] = np.zeros(w_num)
    net['W']['to'] = np.zeros(w_num)

    net['W']['k1'] = net['W']['k3'] = net['W']['d1'] = net['W']['d3'] = rand_in_range([0, 1], w_num)

    net['W']['bp'] = rand_in_range(init_dict['basepair_range'], w_num)
    net['W']['A'] = init_dict['persistence']
    net['W']['L'] = net['W']['bp'] * 0.63e-6
    net['W']['E'] = np.zeros(w_num)
    net['W']['l0'] = np.zeros(w_num)
    net['W']['dist_old'] = np.zeros(w_num)

    w_idx = 0
    for i in range(nodes):
        for j in range(nodes):
            if R[i, j] == 1:
                net['W']['from'][w_idx] = i
                net['W']['to'][w_idx] = j
                p1 = net['P']['states'][i, :2]
                p2 = net['P']['states'][j, :2]
                net['W']['l0'][w_idx] = e_distance(p1, p2)[0]
                w_idx += 1

    net['W']['E'][:] = net['W']['dist_old'][:] = net['W']['l0'][:]
    net['W']['nConnection'] = w_num

    # -------------------------------------------------------------
    # INPUT WEIGHTS (unchanged)
    # -------------------------------------------------------------
    net['W_in'] = net_prev['W_in']

    # -------------------------------------------------------------
    # FEEDBACK WEIGHTS
    # -------------------------------------------------------------
    net['W_fb'] = np.zeros((nodes, init_dict['nOutput']))
    nmax = int(np.ceil(nodes * init_dict['fb_conn']))

    for nI in range(init_dict['nOutput']):
        Idx = random.permutation(nodes)[:nmax]
        net['W_fb'][Idx, nI] = random.uniform(
            init_dict['w_feedback_range'][0],
            init_dict['w_feedback_range'][1],
            nmax
        )

    # -------------------------------------------------------------
    # READOUT WEIGHTS
    # -------------------------------------------------------------
    if init_dict['readout_type'] == 'POSITIONS':
        net['W_out'] = np.zeros((nodes, init_dict['nOutput']))
        nmax = int(nodes * init_dict['out_conn'])

        for nO in range(init_dict['nOutput']):
            Odx = random.permutation(nodes)[:nmax]
            net['W_out'][Odx, nO] = rand_in_range(init_dict['w_out_range'], nmax)

    elif init_dict['readout_type'] == 'LENGTHS':
        w_num = net['W']['l0'].shape[0]
        net['W_out'] = np.zeros((w_num, init_dict['nOutput']))
        nmax = int(np.ceil(w_num * init_dict['out_conn']))

        for nO in range(init_dict['nOutput']):
            Odx = random.permutation(w_num)[:nmax]
            net['W_out'][Odx, nO] = rand_in_range(init_dict['w_out_range'], a=nmax)

    else:
        print("ERROR - unknown output type")

    # -------------------------------------------------------------
    # Final bookkeeping
    # -------------------------------------------------------------
    net['readout_type'] = init_dict['readout_type']
    net['fixed_idx'] = np.where(net['P']['fixed'] == 1)[0]
    net['input_idx'] = np.where(np.sum(net['W_in'], axis=1) != 0)[0]
    net['output_idx'] = np.where(np.sum(net['W_out'], axis=1) != 0)[0]

    net['rk_steps'] = init_dict['rk_steps']

    return net


# ------------------------------------------- #
#              Network all connected          #
#                Non-planar                   #

 
 
def network_system_nonplanar(init_func,parameters):
    net = dict()
    net['init_data'] = init_dict = init_func(parameters)
    num = nodes = init_dict['nodes']
    
    net['P'] = dict()
    
    if init_dict['fixed_positions']:
        net['P']['states'] = init_dict['positions']
    else:
        net['P']['states'] = np.column_stack([
            random.uniform(init_dict['x_lim'][0], init_dict['x_lim'][1], num),
			random.uniform(init_dict['y_lim'][0], init_dict['y_lim'][1], num),
			np.zeros(num),
			np.zeros(num)
        ])
    
    net['P']['force'] = np.zeros((num,2))
    net['P']['fixed'] = np.zeros(num)
    
    net['init_data']['P'] = net['P']
    net['pos_noise'] = 0
    net['dist_noise'] = 0
    
    min_idx = np.argmin(net['P']['states'][:, 0])
    max_idx = np.argmax(net['P']['states'][:, 0])

    if nodes < 3:
        net['P']['fixed'][min_idx] = 1
    else:
        net['P']['fixed'][min_idx] = 1
        net['P']['fixed'][max_idx] = 1
    
    #get random sequence for nodes, generalized.
    names,num_char = get_random_names(num)
    net['names'] = names
    net['num_char'] = num_char
    input_name = parameters['sequence'][:num_char]
    if input_name not in names:
        for name_index in range(len(names)):
            if net['P']['fixed'][name_index] == 0:
                names[name_index] = input_name
                index_of_input_name = name_index
                break
    else:
        index_of_input_name = names.index(input_name)
        if net['P']['fixed'][index_of_input_name] == 1:
            for ijk in range(len(net['P']['fixed'])):
                if net['P']['fixed'][ijk] == 0 and names[ijk] != input_name:
                    names[ijk], names[index_of_input_name] = names[index_of_input_name], names[ijk]
                    index_of_input_name = ijk
                    break
    net['input'] = dict([('name',""), ('index','')])
    net['input']['name'] = input_name
    net['input']['index'] = index_of_input_name
    #get connection sequence from parameter
    connection_string = parameters['sequence']
                    
    
    connection_matrix = np.zeros((nodes,nodes))
    #get fragments of size num_char * 2 to have 2 nodes in that fragment    
    fragments = []
    for i in range(0,len(connection_string),num_char*2):
        fragments.append(connection_string[i:i+num_char*2])
    
    #matrix to see if the connections are already present or not
    established_connections = []
    
    #making connections
    for first_node in range(len(names)):
        for second_node in range(len(names)):
            if first_node != second_node:
                if (first_node,second_node) not in established_connections and (second_node,first_node) not in established_connections:
                    established_connections.append((first_node,second_node))
                    connection_matrix[first_node,second_node] = 1
    print(connection_matrix)
    net['R'] = R = connection_matrix
    
    total_connections = 0
    for row in range(len(connection_matrix)):
        for col in range(len(connection_matrix[row])):
            if connection_matrix[row][col] == 1:
                total_connections += 2
                
    w_num = int(np.sum(R))
    net['W'] = dict()
    net['W']['avgcon'] = total_connections/8
    net['W']['from'] = np.zeros(w_num)
    net['W']['to'] = np.zeros(w_num)
    
    net['W']['k1'] = net['W']['k3'] = net['W']['d1'] = net['W']['d3']  = rand_in_range([0,1], w_num)  
    
    #Initializing Spring Parameters
    net['W']['bp'] = rand_in_range(init_dict['basepair_range'],w_num)
    net['W']['A'] = init_dict['persistence']
    net['W']['L'] = net['W']['bp']*0.63e-9#np.array([12e-9])#
    net['W']['E'] = np.zeros(w_num) #np.array([5e-9]*w_num#np.sqrt(net['W']['L']*4e-9)
    net['W']['l0'] = np.zeros(w_num) #The initial length of springs, same size as number of connections
    net['W']['dist_old'] = np.zeros(w_num) #The old distance, i.e. distance at t-delta(t) seconds.same size as number of connections
    
    w_idx  = 0
    
    for i in range(init_dict['nodes']):
        for j in range(init_dict['nodes']):
            if R[i, j] == 1:
                net['W']['from'][w_idx] = i
                net['W']['to'][w_idx] = j
                p1 =net['P']['states'][i, :2]
                p2 =net['P']['states'][j,:2]
                net['W']['l0'][w_idx] = e_distance(p1,p2)[0]
                w_idx += 1
 
    net['W']['E'][:] = net['W']['dist_old'][:] = net['W']['l0'][:]
    net['W_in'] = np.zeros((init_dict['nodes'], init_dict['nInput']))
    nmax = int(init_dict['nodes'] * init_dict['input_percentage'])
     
    net['W_in'][index_of_input_name] = random.uniform(init_dict['w_input_range'][0], init_dict['w_input_range'][1])
    
    
    Wins = []
    fixeds = []
    for win in net['W_in']:
        Wins.append(float(win))
    
    
    for pxs in net['P']['fixed']:
        fixeds.append(int(pxs))

    Wins = np.array(Wins)
    fixeds = np.array(fixeds)
    
    if not (Wins.any() > 0.):
        argmin = np.argmin(fixeds)
        net['W_in'][argmin] = random.uniform(init_dict['w_input_range'][0], init_dict['w_input_range'][1], 1)
    
    net['W_fb'] = np.zeros((init_dict['nodes'], init_dict['nOutput']))
    
    nmax = int(np.ceil(init_dict['nodes'] * init_dict['fb_conn']))
    
    for nI in range(init_dict['nOutput']):
        Idx = random.permutation(init_dict['nodes'])
        Idx = Idx[:nmax]
        net['W_fb'][Idx,nI] = random.uniform(init_dict['w_feedback_range'][0], init_dict['w_feedback_range'][1], nmax)
    
    if init_dict['readout_type'] == 'POSITIONS':
        net['W_out'] = np.zeros((init_dict['nodes'], init_dict['nOutput']))
        nmax = int(init_dict['nodes']*init_dict['out_conn'])
        
        for nO in range(0,init_dict['nOutput']):
            Odx = random.permutation(init_dict['nodes']) #random input connections
            Odx[nmax+1:] = []
            net['W_out'][Odx,nO] = rand_in_range(init_dict['w_out_range'], nmax) #between -iScale and +iScale
    
    else:
        if init_dict['readout_type'] == 'LENGTHS':
            w_num = (net['W']['l0']).shape[0]
            net['W_out'] = np.zeros((w_num, init_dict['nOutput']))
            nmax = int(np.ceil(w_num * init_dict['out_conn']))
            
            for nO in range(init_dict['nOutput']):
                Odx = random.permutation(w_num)
                Odx[nmax+1:] = []
                net['W_out'][Odx, nO] = rand_in_range(init_dict['w_out_range'], a=nmax)
        
        else:
            print('ERROR - unknown output chosen')
    
    net['readout_type'] = init_dict['readout_type']
    net['fixed_idx'] = np.where(net['P']['fixed'] == 1)[0]
    #print(net['fixed_idx'])
    net['input_idx'] = np.where(np.sum(net['W_in'], axis=1) != 0)[0]
    net['output_idx'] = np.where(np.sum(net['W_out'], axis=1) != 0)[0]
    
    net['rk_steps'] = net['init_data']['rk_steps']
    
    net['W']['nConnection'] = net['W']['k1'].shape[0]
	
    
    return net


def network_change_nonplanar(net_prev,sequence):
    if isinstance(sequence,list):
        sequence = sequence[0]
    
    net = dict()
    net['init_data'] = init_dict = net_prev['init_data']
    num = nodes = init_dict['nodes']
    
    net['P'] = net_prev['P']
    
    net['P']['force'] = np.zeros((num,2))
    net['P']['fixed'] = np.zeros(num)
    
    net['init_data']['P'] = net['P']
    net['pos_noise'] = 0
    net['dist_noise'] = 0
    
    min_idx = np.argmin(net['P']['states'][:, 0])
    max_idx = np.argmax(net['P']['states'][:, 0])

    if nodes < 3:
        net['P']['fixed'][min_idx] = 1
    else:
        net['P']['fixed'][min_idx] = 1
        net['P']['fixed'][max_idx] = 1
    
    #get random sequence for nodes, generalized.
    _,num_char = get_random_names(num)
    names = net_prev['names']
    net['names'] = names
    net['num_char'] = num_char
    net['input'] = dict([('name',""), ('index','')])
    index_of_input_name = net['input']['index'] = net_prev['input']['index']
    
    #get connection sequence from parameter
    connection_string = sequence
    
    connection_matrix = np.zeros((nodes,nodes))
    #get fragments of size num_char * 2 to have 2 nodes in that fragment    
    fragments = []
    for i in range(0,len(connection_string),num_char*2):
        fragments.append(connection_string[i:i+num_char*2])
    
    #matrix to see if the connections are already present or not
    established_connections = []
    
    for first_node in range(len(names)):
        for second_node in range(len(names)):
            if first_node != second_node:
                if (first_node,second_node) not in established_connections and (second_node,first_node) not in established_connections:
                    established_connections.append((first_node,second_node))
                    connection_matrix[first_node,second_node] = 1
    
    print(connection_matrix)
    net['R'] = R = connection_matrix
    w_num = int(np.sum(R))
    net['W'] = dict()
    net['W']['avgcon'] = np.sum(R)/8
    net['W']['from'] = np.zeros(w_num)
    net['W']['to'] = np.zeros(w_num)
    
    net['W']['k1'] = net['W']['k3'] = net['W']['d1'] = net['W']['d3']  = rand_in_range([0,1], w_num)  
    
    """
    
    Now we define the spring parameters.
    
        'bp': base-pairs of springs
        'A':  persistence lengths
        'L':  contour lengths, defined as 
                Lc = 0.63 * bp nm
                by default we use bp, and assume it is of order 10^3. so we directly use Lc=0.63*bp micron
        'E': Spring lengths at t=0
        'l0':Spring lengths at t=0
    
    """
    net['W']['bp'] = rand_in_range(init_dict['basepair_range'],w_num)
    net['W']['A'] = init_dict['persistence']
    net['W']['L'] = net['W']['bp']*0.63e-6
    net['W']['E'] = np.zeros(w_num)
    net['W']['l0'] = np.zeros(w_num)
    
    net['W']['dist_old'] = np.zeros(w_num)
    
    w_idx  = 0
    
    for i in range(init_dict['nodes']):
        for j in range(init_dict['nodes']):
            if R[i, j] == 1:
                net['W']['from'][w_idx] = i
                net['W']['to'][w_idx] = j
                p1 =net['P']['states'][i, :2]
                p2 =net['P']['states'][j,:2]
                net['W']['l0'][w_idx] = e_distance(p1,p2)[0]
                w_idx += 1
 
    net['W']['E'][:] = net['W']['dist_old'][:] = net['W']['l0'][:]
    net['W_in'] = np.zeros((init_dict['nodes'], init_dict['nInput']))
    nmax = int(init_dict['nodes'] * init_dict['input_percentage'])
    
    net['W_in'][index_of_input_name] = random.uniform(init_dict['w_input_range'][0], init_dict['w_input_range'][1])
    
    
    Wins = []
    fixeds = []
    for win in net['W_in']:
        Wins.append(float(win))
    
    for pxs in net['P']['fixed']:
        fixeds.append(int(pxs))

    Wins = np.array(Wins)
    fixeds = np.array(fixeds)
    
    net['W_in'] = net_prev['W_in']

    net['W_fb'] = np.zeros((init_dict['nodes'], init_dict['nOutput']))
    
    nmax = int(np.ceil(init_dict['nodes'] * init_dict['fb_conn']))
    
    for nI in range(init_dict['nOutput']):
        Idx = random.permutation(init_dict['nodes'])
        Idx = Idx[:nmax]
        net['W_fb'][Idx,nI] = random.uniform(init_dict['w_feedback_range'][0], init_dict['w_feedback_range'][1], nmax)
    
    if init_dict['readout_type'] == 'POSITIONS':
        net['W_out'] = np.zeros((init_dict['nodes'], init_dict['nOutput']))
        nmax = int(init_dict['nodes']*init_dict['out_conn'])
        
        for nO in range(0,init_dict['nOutput']):
            Odx = random.permutation(init_dict['nodes']) #random input connections
            Odx[nmax+1:] = []
            net['W_out'][Odx,nO] = rand_in_range(init_dict['w_out_range'], nmax) #between -iScale and +iScale
    
    else:
        if init_dict['readout_type'] == 'LENGTHS':
            w_num = (net['W']['l0']).shape[0]
            net['W_out'] = np.zeros((w_num, init_dict['nOutput']))
            nmax = int(np.ceil(w_num * init_dict['out_conn']))
            
            for nO in range(init_dict['nOutput']):
                Odx = random.permutation(w_num)
                Odx[nmax+1:] = []
                net['W_out'][Odx, nO] = rand_in_range(init_dict['w_out_range'], a=nmax)
        
        else:
            print('ERROR - unknown output chosen')
    
    #Store the readout type
    net['readout_type'] = init_dict['readout_type']
    
    #Store the fixed node index
    net['fixed_idx'] = np.where(net['P']['fixed'] == 1)[0]
    
    #Store the input node index
    net['input_idx'] = np.where(np.sum(net['W_in'], axis=1) != 0)[0]
    
    #Store the output node index
    net['output_idx'] = np.where(np.sum(net['W_out'], axis=1) != 0)[0]
    
    #Store the runge kutta step
    net['rk_steps'] = net['init_data']['rk_steps']
    
    #Store number of connections
    net['W']['nConnection'] = net['W']['k1'].shape[0]
	
 
    return net

