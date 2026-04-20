"""
This code was adapted from Hauser et al. where the springs are realized as DNA-strands. 
This file simulates the DNA spring-bead reservoir system, and as an output gives the simulation matrix.
"""

import numpy as np
import copy
import time as time

# Importing functions
from functions.wlc import dna_spring
from functions.e_distance import e_distance
from functions.ode import ode

def simulate_dna_spring_reservoir(network,input_force,damping_coeff):
    
    """
    Simulate the time evolution of a DNA-based spring network as a reservoir computer.

    This function evolves a system of particles (nodes) connected via DNA-inspired springs
    (modeled with a nonlinear worm-like chain force), subject to input forces, damping, and
    network connectivity. The simulation integrates the motion using a custom ODE solver for
    each particle in both x and y dimensions.

    Args:
        network (dict): A dictionary defining the initial state of the reservoir system. Must contain:
            - 'P': Node states (position, velocity, force)
            - 'W': Network structure (connections, spring constants, rest lengths, base pairs)
            - 'W_in': Input weight matrix
            - 'W_out': Readout weight matrix
            - 'fixed_idx': Indices of fixed nodes
            - 'init_data': Simulation parameters (time_step, show_steps, number of nodes)
            - 'readout_type': One of {'POSITIONS', 'LENGTHS'}
        input_force (np.ndarray): 1D array of external input forces applied over time.
        damping_coeff (float): Damping coefficient to simulate drag/friction.

    Returns:
        tuple:
            - net_after (dict): The updated network state after simulation (with evolved positions and internal states).
            - sim_data (dict): A dictionary of simulation outputs, including:
                - 'Fx', 'Fy': Forces on each node
                - 'Fdampx', 'Fdampy': Damping forces
                - 'Fspring': Force from each spring
                - 'Sx', 'Sy': Position trajectories
                - 'Sxd', 'Syd': Velocities
                - 'D': Spring lengths
                - 'O': Output signal (from LENGTHS or POSITIONS)
                - 'bead_positions', 'bead_velocities': Complete 2D trajectories
                - 'output': Raw spring extensions over time

    Notes:
        - This simulation uses a second-order ODE to integrate motion.
        - Handles both fixed and mobile nodes.
        - Unstable simulations are detected via NaN outputs and halted early.
    """
    
    net = copy.deepcopy(network)
    time_step = net['init_data']['time_step']
    show_steps = net['init_data']['show_steps']

    # Getting number of nodes
    num = net['init_data']['nodes']
    in_idx = np.where(net['W_in'] != 0)[0]
    # Getting number of inputs
    input = input_force
   
    len_i = len(input_force)
    sim_time = len_i * time_step

    P = net['P']  # Points information
    W = net['W']
    
    fix_idx = net['fixed_idx']

    P0 = copy.deepcopy(P)
    sim_data = {
        'Fx': np.zeros((len_i, num)),  # Forces in x-dimension
        'Fy': np.zeros((len_i, num)),  # Forces in y-dimension
        'Fdampx':  np.zeros((len_i, num)), #Fdrag 
        'Fdampy':  np.zeros((len_i, num)), #Fdrag 
        'Fspring': np.zeros((len_i, W['k1'].shape[0])), #Spring Force
        'Sx': np.zeros((len_i, num)),  # Length in simulation in x-dimension
        'Sy': np.zeros((len_i, num)),  # Length in simulation in y-dimension
        'Sxd': np.zeros((len_i, num)),  # Velocity in simulation in x-dimension
        'Syd': np.zeros((len_i, num)),  # Velocity in simulation in y-dimension
        'Sx_off': np.zeros((len_i, num)),  # The offset in matrix
        'P': np.zeros((len_i,num)),
        'D': np.zeros((len_i,W['k1'].shape[0])),  # Internal state
        #'forces': np.zeros((len_i, W['k1'].shape[0])),
        'contours': net['W']['L'],
        'velocities': np.zeros((len_i, num)),
        'extension': np.zeros((int(W['nConnection']), len_i)),
        'bead_positions':np.zeros((len_i,num,2)),
        'bead_velocities':np.zeros((len_i,num,2))
    }
    
    if net['readout_type'] == 'POSITIONS':
        sim_data['O'] = np.zeros((len_i,num))
    if net['readout_type'] == 'LENGTHS':
        sim_data['O'] = np.zeros((len_i,W['k1'].shape[0]))
    
    try:
        hvalue = time_step/net['rk_steps']
    except:
        hvalue = 0
    
    #START THE Simulation LOOP
    for idx in range(len_i):
        #Setting all the old forces to be zero
        P['force'][:,:2] = np.zeros((num,2))
        
        #Iterate through all springs and find spring forces
        for c in range(int(W['nConnection'])):
            fr = int(W['from'][c])
            to = int(W['to'][c]) 
            
            p_from = np.array(P['states'][fr][0:2])
            p_to = np.array(P['states'][to][0:2])
            d,ndir = e_distance(p1=p_from, p2=p_to)
            
            sim_data['D'][idx,c] = d
            
            R = W['E'][c]
            bp = W['bp'][c]
            
            extension = d - R
            
            #Calculate force by the spring due to extension
            Fspring = dna_spring(z=extension,basepair=bp)
            
            sim_data['extension'][c,idx] = extension
            sim_data['Fspring'][idx,c] = Fspring
            
            
            Fx = Fspring*ndir[0]
            Fy = Fspring*ndir[1]
            
            if(int(P['fixed'][to]) == 0):
                P['force'][to][0] += Fx
                P['force'][to][1] += Fy
                sim_data['Fx'][idx,to] = +1*Fx    
                sim_data['Fy'][idx,to] = +1*Fy
            
            
            if(int(P['fixed'][fr]) == 0):
                P['force'][fr][0] -= Fx
                P['force'][fr][1] -= Fy
                sim_data['Fx'][idx,fr] = -1*Fx    
                sim_data['Fy'][idx,fr] = -1*Fy
            
            W['dist_old'][c] = d
            
            #Now, we have done the DNA-spring length simulation.
            
        #Now we proceed to calculate the bead position, velocities
        
        for p in range(num):
            
            #Get all the states on the points
            states = P['states'][p]
            
            #Get all the forces on the particle
            Fx = fx = P['force'][p,0]
            Fy = fy = P['force'][p,1]
            
            #first we calculate difference of position of particle in last two steps
            
            #If the particle is fixed, set its forces to be zero.
            if p in fix_idx:
                P['states'][p][2:4] = 0
                P['force'][p][0:2] = 0
                typn = 'fixed node'

            dvx = states[2]
            dvy = states[3]
            
            if p in in_idx:
                Fx += input_force[idx]
            
            #Update the forces in the simulation dictionary
            P['force'][p,0] = Fx
            P['force'][p,1] = Fy
            sim_data['Fx'][idx,p] = Fx
            sim_data['Fy'][idx,p] = Fy
            
            #Calculate the new position of the particle.
            xx,vx,dampx = ode(x0 = states[0],xd0=states[2],F=Fx, bv=damping_coeff,dt=time_step)#runge(a=0,b=time_step,x0 = states[0], h= hvalue, F= Fx, bv=damping_coeff,xd0 = dvx)
            xy,vy,dampy = ode(x0 = states[1],xd0=states[3],F=Fy, bv=damping_coeff,dt=time_step)#runge(a=0,b=time_step,x0 = states[1], h= hvalue, F= Fy, bv=damping_coeff,xd0 = dvy)
            
            #Update the damping force in the simulation dictionary
            sim_data['Fdampx'][idx,p] = dampx
            sim_data['Fdampy'][idx,p] = dampy
            P['states'][p,0] = xx
            P['states'][p,1] = xy
            P['states'][p,2] = vx
            P['states'][p,3] = vy
            
            #Update the positions in the simulation dictionary
            sim_data['Sx'][idx,p] = xx
            sim_data['Sy'][idx,p] = xy
            sim_data['Sxd'][idx,p] = vx
            sim_data['Syd'][idx,p] = vy
            sim_data['bead_positions'][idx,p] = [float(xx),float(xy)]
            sim_data['bead_velocities'][idx,p] = [float(vx),float(vy)]
            

        #Simulation loop done.
        
        
        #Check if there is any problem, if there is, break the simulation and return the parameters.
        
        if np.count_nonzero(np.isnan(sim_data['O'][idx])) > 0:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print("UNSTABLE              SIMULATION")
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            return net, sim_data
    
    #Setting Output   
    if net['readout_type'] == 'LENGTHS':
        sim_data['O'] = sim_data['D'] @ net['W_out']
    else:
        sim_data['O'] = sim_data['Sx'] @ net['W_out']   
    net_after = copy.deepcopy(net)
    net_after['P'] = P
    net_after['W'] = W
    
    #Save spring lengths as output (without weighted sum)
    l1 = np.zeros_like(sim_data['D'])
    for row in range(len(sim_data['D'])):
        for col in range(len(sim_data['D'][row])):
            l1[row][col] = sim_data['D'][row][col]
    sim_data['output'] = l1
    return net_after, sim_data


def simulate_dna_spring_reservoir_brownian(network,input_force,damping_coeff):
    
    """
    Simulate the time evolution of a DNA-based spring network as a reservoir computer.

    This function evolves a system of particles (nodes) connected via DNA-inspired springs
    (modeled with a nonlinear worm-like chain force), subject to input forces, damping, and
    network connectivity. The simulation integrates the motion using a custom ODE solver for
    each particle in both x and y dimensions.

    Args:
        network (dict): A dictionary defining the initial state of the reservoir system. Must contain:
            - 'P': Node states (position, velocity, force)
            - 'W': Network structure (connections, spring constants, rest lengths, base pairs)
            - 'W_in': Input weight matrix
            - 'W_out': Readout weight matrix
            - 'fixed_idx': Indices of fixed nodes
            - 'init_data': Simulation parameters (time_step, show_steps, number of nodes)
            - 'readout_type': One of {'POSITIONS', 'LENGTHS'}
        input_force (np.ndarray): 1D array of external input forces applied over time.
        damping_coeff (float): Damping coefficient to simulate drag/friction.

    Returns:
        tuple:
            - net_after (dict): The updated network state after simulation (with evolved positions and internal states).
            - sim_data (dict): A dictionary of simulation outputs, including:
                - 'Fx', 'Fy': Forces on each node
                - 'Fdampx', 'Fdampy': Damping forces
                - 'Fspring': Force from each spring
                - 'Sx', 'Sy': Position trajectories
                - 'Sxd', 'Syd': Velocities
                - 'D': Spring lengths
                - 'O': Output signal (from LENGTHS or POSITIONS)
                - 'bead_positions', 'bead_velocities': Complete 2D trajectories
                - 'output': Raw spring extensions over time

    Notes:
        - This simulation uses a second-order ODE to integrate motion.
        - Handles both fixed and mobile nodes.
        - Unstable simulations are detected via NaN outputs and halted early.
    """
    
    net = copy.deepcopy(network)
    time_step = net['init_data']['time_step']
    show_steps = net['init_data']['show_steps']

    # Getting number of nodes
    num = net['init_data']['nodes']
    in_idx = np.where(net['W_in'] != 0)[0]
    # Getting number of inputs
    input = input_force
   
    len_i = len(input_force)
    sim_time = len_i * time_step

    P = net['P']  # Points information
    W = net['W']
    
    fix_idx = net['fixed_idx']

    P0 = copy.deepcopy(P)
    sim_data = {
        'Fx': np.zeros((len_i, num)),  # Forces in x-dimension
        'Fy': np.zeros((len_i, num)),  # Forces in y-dimension
        'Fdampx':  np.zeros((len_i, num)), #Fdrag 
        'Fdampy':  np.zeros((len_i, num)), #Fdrag 
        'Fspring': np.zeros((len_i, W['k1'].shape[0])), #Spring Force
        'Sx': np.zeros((len_i, num)),  # Length in simulation in x-dimension
        'Sy': np.zeros((len_i, num)),  # Length in simulation in y-dimension
        'Sxd': np.zeros((len_i, num)),  # Velocity in simulation in x-dimension
        'Syd': np.zeros((len_i, num)),  # Velocity in simulation in y-dimension
        'Sx_off': np.zeros((len_i, num)),  # The offset in matrix
        'P': np.zeros((len_i,num)),
        'D': np.zeros((len_i,W['k1'].shape[0])),  # Internal state
        #'forces': np.zeros((len_i, W['k1'].shape[0])),
        'contours': net['W']['L'],
        'velocities': np.zeros((len_i, num)),
        'extension': np.zeros((int(W['nConnection']), len_i)),
        'bead_positions':np.zeros((len_i,num,2)),
        'bead_velocities':np.zeros((len_i,num,2))
    }
    
    if net['readout_type'] == 'POSITIONS':
        sim_data['O'] = np.zeros((len_i,num))
    if net['readout_type'] == 'LENGTHS':
        sim_data['O'] = np.zeros((len_i,W['k1'].shape[0]))
    
    try:
        hvalue = time_step/net['rk_steps']
    except:
        hvalue = 0
    
    #START THE Simulation LOOP
    for idx in range(len_i):
        #Setting all the old forces to be zero
        P['force'][:,:2] = np.zeros((num,2))
        
        #Iterate through all springs and find spring forces
        for c in range(int(W['nConnection'])):
            fr = int(W['from'][c])
            to = int(W['to'][c]) 
            
            p_from = np.array(P['states'][fr][0:2])
            p_to = np.array(P['states'][to][0:2])
            d,ndir = e_distance(p1=p_from, p2=p_to)
            
            sim_data['D'][idx,c] = d
            
            R = W['E'][c]
            bp = W['bp'][c]
            
            extension = d - R
            
            #Calculate force by the spring due to extension
            Fspring = dna_spring(z=extension,basepair=bp)
            
            sim_data['extension'][c,idx] = extension
            sim_data['Fspring'][idx,c] = Fspring
            
            
            Fx = Fspring*ndir[0]
            Fy = Fspring*ndir[1]
            
            if(int(P['fixed'][to]) == 0):
                P['force'][to][0] += Fx
                P['force'][to][1] += Fy
                sim_data['Fx'][idx,to] = +1*Fx    
                sim_data['Fy'][idx,to] = +1*Fy
            
            
            if(int(P['fixed'][fr]) == 0):
                P['force'][fr][0] -= Fx
                P['force'][fr][1] -= Fy
                sim_data['Fx'][idx,fr] = -1*Fx    
                sim_data['Fy'][idx,fr] = -1*Fy
            
            W['dist_old'][c] = d
            
            #Now, we have done the DNA-spring length simulation.
            
        #Now we proceed to calculate the bead position, velocities
        
        for p in range(num):
            
            #Get all the states on the points
            states = P['states'][p]
            
            #Get all the forces on the particle
            Fx = fx = P['force'][p,0]
            Fy = fy = P['force'][p,1]
            
            #first we calculate difference of position of particle in last two steps
            
            #If the particle is fixed, set its forces to be zero.
            if p in fix_idx:
                P['states'][p][2:4] = 0
                P['force'][p][0:2] = 0
                typn = 'fixed node'

            dvx = states[2]
            dvy = states[3]
            
            if p in in_idx:
                Fx += input_force[idx]
            
            # Add Brownian noise BEFORE storing the force
            kT = 4.114e-21  # Joules (300 K)
            sigma = np.sqrt(2 * damping_coeff * kT / time_step)

            Fx += np.random.normal(0, sigma)
            Fy += np.random.normal(0, sigma)

            # Now store the full (spring + input + noise) force
            P['force'][p,0] = Fx
            P['force'][p,1] = Fy
            
            sim_data['Fx'][idx,p] = Fx
            sim_data['Fy'][idx,p] = Fy

            
            #Calculate the new position of the particle.
            xx,vx,dampx = ode(x0 = states[0],xd0=states[2],F=Fx, bv=damping_coeff,dt=time_step)#runge(a=0,b=time_step,x0 = states[0], h= hvalue, F= Fx, bv=damping_coeff,xd0 = dvx)
            xy,vy,dampy = ode(x0 = states[1],xd0=states[3],F=Fy, bv=damping_coeff,dt=time_step)#runge(a=0,b=time_step,x0 = states[1], h= hvalue, F= Fy, bv=damping_coeff,xd0 = dvy)
            
            #Update the damping force in the simulation dictionary
            sim_data['Fdampx'][idx,p] = dampx
            sim_data['Fdampy'][idx,p] = dampy
            P['states'][p,0] = xx
            P['states'][p,1] = xy
            P['states'][p,2] = vx
            P['states'][p,3] = vy
            
            #Update the positions in the simulation dictionary
            sim_data['Sx'][idx,p] = xx
            sim_data['Sy'][idx,p] = xy
            sim_data['Sxd'][idx,p] = vx
            sim_data['Syd'][idx,p] = vy
            sim_data['bead_positions'][idx,p] = [float(xx),float(xy)]
            sim_data['bead_velocities'][idx,p] = [float(vx),float(vy)]
            

        #Simulation loop done.
        
        
        #Check if there is any problem, if there is, break the simulation and return the parameters.
        
        if np.count_nonzero(np.isnan(sim_data['O'][idx])) > 0:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print("UNSTABLE              SIMULATION")
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            return net, sim_data
    
    #Setting Output   
    if net['readout_type'] == 'LENGTHS':
        sim_data['O'] = sim_data['D'] @ net['W_out']
    else:
        sim_data['O'] = sim_data['Sx'] @ net['W_out']   
    net_after = copy.deepcopy(net)
    net_after['P'] = P
    net_after['W'] = W
    
    #Save spring lengths as output (without weighted sum)
    l1 = np.zeros_like(sim_data['D'])
    for row in range(len(sim_data['D'])):
        for col in range(len(sim_data['D'][row])):
            l1[row][col] = sim_data['D'][row][col]
    sim_data['output'] = l1
    return net_after, sim_data