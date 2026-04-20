"""
Volterra (Task-1) Evaluation for DNA Spring-Based Reservoir Networks

This file defines functions to evaluate the performance of a DNA-based mechanical 
reservoir computing network on the Volterra series prediction task. It loads task-specific 
input/output time series, runs the simulation using a DNA spring network, fits the 
readout weights using linear regression, and returns error metrics such as MSE and NMSE.

Functions:
    - change_connection: Updates the DNA-based network connectivity using a new sequence.
    - test_volterra_genetic_all_params: Simulates the reservoir, trains readout, tests performance.
    - test_volterra_1: Wrapper to return either NMSE or full evaluation output.

Requires:
    - `datasets/Task1.mat`: Contains the input-output data for the Volterra task.
    - `simulate_dna_spring_reservoir`: Simulates dynamics of the network with given input.
    - `network_change`: Alters the network based on a new sequence.
"""


import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from system import network_change
from simulate_dna_beads import simulate_dna_spring_reservoir,simulate_dna_spring_reservoir_brownian
from functions.mse import nmse_calculate,mean_squared_error

def change_connection(net, sequence):
    
    """
    Updates the network's internal connectivity using a new DNA sequence.

    Args:
        net (dict): Original DNA spring network dictionary.
        sequence (str): DNA sequence used to define new connections.

    Returns:
        dict: Updated network dictionary with modified connections.
    """
    
    og_net = net
    net = network_change(og_net,sequence)
    return net

def test_volterra_genetic_all_params(network, sequence):
    """
    TASK-1
    Evaluates a DNA-based spring reservoir network on the Volterra task using a given DNA sequence.

    The function simulates the network with training inputs, learns output weights using 
    linear regression, and tests on unseen data to calculate error metrics.

    Args:
        network (dict): The initialized DNA spring network.
        sequence (str): DNA sequence to define spring connections.

    Returns:
        tuple:
            - mse (float): Mean squared error on the test set.
            - net_initial (dict): Network before any simulation.
            - net_after_inp (dict): Network after training input simulation.
            - net_test_out (dict): Network with trained output weights used for testing.
            - sim_data (dict): Simulation data during training phase.
            - sim_data_test (dict): Simulation data during testing phase.
            - nmse (float): Normalized mean squared error.
    """
    net = change_connection(network,sequence)
    
    try:
        data = loadmat('datasets/Task1.mat')
    except FileNotFoundError:
        print("Error: File 'datasets/Task1.mat' not found.")
        exit(1)
    
    wash_out = 0
    start = 0#80000 - wash_out
    len_data = 15000#250000 + start #250000 + start
    len_test = 15000
    factor = 1e-11
    U = np.array(data['dat']['u'][0][0][start:len_data]) * factor
    Y = np.array(data['dat']['y'][0][0][start:len_data]) * factor

    U_test = np.array(data['dat']['u'][0][0][len_data:len_data + len_test]) * factor
    Y_test = np.array(data['dat']['y'][0][0][len_data:len_data + len_test]) * factor

    scaler = StandardScaler()
    
    bvalue = 1.67e-7
    net2, sim_data = simulate_dna_spring_reservoir(net, U, bvalue)
    
    if net['readout_type'] == 'LENGTHS':
        X = sim_data['D'][wash_out:, :]
    else:
        X = sim_data['Sx'][wash_out:, :]
    Yw = Y[wash_out:]
    Yw = Yw.ravel()
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, Yw)
    W_out = lr.coef_

    net_test = net2
    net_test['W_out'] = np.array(W_out)
    o = X @ W_out

    net_test_out, sim_data_test = simulate_dna_spring_reservoir(net_test, U_test, bvalue)
    Ysim = sim_data_test['O']

    Y_test_mse = Y_test.ravel() / factor
    Ysim_mse = Ysim / factor

    
    mse = mean_squared_error(Y_test_mse, Ysim_mse)
    
    tmean = np.mean(Y_test_mse)
    smean = np.mean(Ysim_mse)
    
    nmse = nmse_calculate(predicted=Y_test_mse,measured=Ysim_mse)
    
    return float(mse),net,net2,net_test_out, sim_data, sim_data_test,nmse

def test_volterra_1(network,sequence,allparams=False):
    """
    Wrapper for evaluating the DNA-based reservoir on the Volterra Task-1.

    Args:
        network (dict): The initialized DNA spring network.
        sequence (str): DNA sequence for connection reconfiguration.
        allparams (bool): If True, returns all outputs from test_volterra_genetic_all_params;
                          otherwise, returns only NMSE.

    Returns:
        float or tuple: Either NMSE (if allparams=False), or full output tuple from 
                        `test_volterra_genetic_all_params` (if allparams=True).
    """
    mse,net_initial,net_after_inp, net_test_out, sim_data, sim_data_test,nmse = test_volterra_genetic_all_params(network,sequence)
    
    if allparams:
        return mse,net_initial,net_after_inp, net_test_out, sim_data, sim_data_test,nmse
    else:
        return nmse



#-------------------------------#
#################################
### Task1 Custom net function ###
#################################
#-------------------------------#

def test_volterra_genetic_all_params_custom(network, sequence,network_change):
    net = change_connection(network,sequence,network_change)
    
    try:
        data = loadmat('datasets/Task1.mat')
    except FileNotFoundError:
        print("Error: File 'datasets/Task1.mat' not found.")
        exit(1)
    
    wash_out = 80000
    start = 80000 - wash_out
    len_data = 250000 + start #250000 + start
    len_test = 15000
    factor = 1e-11
    U = np.array(data['dat']['u'][0][0][start:len_data]) * factor
    Y = np.array(data['dat']['y'][0][0][start:len_data]) * factor

    U_test = np.array(data['dat']['u'][0][0][len_data:len_data + len_test]) * factor
    Y_test = np.array(data['dat']['y'][0][0][len_data:len_data + len_test]) * factor

    scaler = StandardScaler()
    
    bvalue = 1.67e-7
    net2, sim_data = simulate_dna_spring_reservoir(net, U, bvalue)
    
    if net['readout_type'] == 'LENGTHS':
        X = sim_data['D'][wash_out:, :]
    else:
        X = sim_data['Sx'][wash_out:, :]
    
    print(X)
    Yw = Y[wash_out:]
    Yw = Yw.ravel()
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, Yw)
    W_out = lr.coef_

    net_test = net2
    net_test['W_out'] = np.array(W_out)
    o = X @ W_out

    net_test_out, sim_data_test = simulate_dna_spring_reservoir(net_test, U_test, bvalue)
    Ysim = sim_data_test['O']
    print(Ysim)
    Y_test_mse = Y_test.ravel() / factor
    Ysim_mse = Ysim / factor

    
    mse = mean_squared_error(Y_test_mse, Ysim_mse)
    
    tmean = np.mean(Y_test_mse)
    smean = np.mean(Ysim_mse)
    
    nmse = nmse_calculate(predicted=Y_test_mse,measured=Ysim_mse)
    
    return float(mse),net,net2,net_test_out, sim_data, sim_data_test,nmse,tmean,smean

def test_volterra_1_custom(network,sequence,network_change,allparams=False):
    mse,net_initial,net_after_inp, net_test_out, sim_data, sim_data_test,nmse,tmean,smean = test_volterra_genetic_all_params_custom(network,sequence,network_change)
    
    if allparams:
        return mse,net_initial,net_after_inp, net_test_out, sim_data, sim_data_test,nmse,tmean,smean
    else:
        return nmse



#---------------------------------------#
#########################################
#### BROWNIAN MOTION ADDED TASK EVAL ####
#########################################
#---------------------------------------#

def test_volterra_genetic_all_params_brownian(network, sequence):
    """
    TASK-1
    Evaluates a DNA-based spring reservoir network on the Volterra task using a given DNA sequence.

    The function simulates the network with training inputs, learns output weights using 
    linear regression, and tests on unseen data to calculate error metrics.

    Args:
        network (dict): The initialized DNA spring network.
        sequence (str): DNA sequence to define spring connections.

    Returns:
        tuple:
            - mse (float): Mean squared error on the test set.
            - net_initial (dict): Network before any simulation.
            - net_after_inp (dict): Network after training input simulation.
            - net_test_out (dict): Network with trained output weights used for testing.
            - sim_data (dict): Simulation data during training phase.
            - sim_data_test (dict): Simulation data during testing phase.
            - nmse (float): Normalized mean squared error.
    """
    net = change_connection(network,sequence)
    
    try:
        data = loadmat('datasets/Task1.mat')
    except FileNotFoundError:
        print("Error: File 'datasets/Task1.mat' not found.")
        exit(1)
    
    wash_out = 80000
    start = 80000 - wash_out
    len_data = 250000 + start #250000 + start
    len_test = 15000
    factor = 1e-11
    U = np.array(data['dat']['u'][0][0][start:len_data]) * factor
    Y = np.array(data['dat']['y'][0][0][start:len_data]) * factor

    U_test = np.array(data['dat']['u'][0][0][len_data:len_data + len_test]) * factor
    Y_test = np.array(data['dat']['y'][0][0][len_data:len_data + len_test]) * factor

    scaler = StandardScaler()
    
    bvalue = 1.67e-7
    net2, sim_data = simulate_dna_spring_reservoir_brownian(net, U, bvalue)
    
    if net['readout_type'] == 'LENGTHS':
        X = sim_data['D'][wash_out:, :]
    else:
        X = sim_data['Sx'][wash_out:, :]
    Yw = Y[wash_out:]
    Yw = Yw.ravel()
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, Yw)
    W_out = lr.coef_

    net_test = net2
    net_test['W_out'] = np.array(W_out)
    o = X @ W_out

    net_test_out, sim_data_test = simulate_dna_spring_reservoir_brownian(net_test, U_test, bvalue)
    Ysim = sim_data_test['O']

    Y_test_mse = Y_test.ravel() / factor
    Ysim_mse = Ysim / factor

    
    mse = mean_squared_error(Y_test_mse, Ysim_mse)
    
    tmean = np.mean(Y_test_mse)
    smean = np.mean(Ysim_mse)
    
    nmse = nmse_calculate(predicted=Y_test_mse,measured=Ysim_mse)
    
    return nmse