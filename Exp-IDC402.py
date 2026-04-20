"""
CRITICALITY ANALYSIS PIPELINE
"""

from system import *
from functions.random_connections import change_to_random_connections
from Task1 import test_volterra_1
import pandas as pd
import random, os, psutil, copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.io import loadmat
from simulate_dna_beads import simulate_dna_spring_reservoir

process = psutil.Process(os.getpid())

# -----------------------------
# ANALYSIS FUNCTIONS
# -----------------------------

def compute_damage(sim1, sim2):
    return np.mean(np.abs(sim1['D'] - sim2['D']), axis=1)


def compute_lyapunov(D, dt):
    eps = 1e-12
    logD = np.log(D + eps)

    start = 100
    T_fit = min(1000, len(D)//2)

    t = np.arange(T_fit-start) * dt
    slope = np.polyfit(t, logD[start:T_fit], 1)[0]
    return slope


def compute_powerlaw_exponent(D):
    eps = 1e-12
    t = np.arange(len(D))
    
    valid = (D > eps)
    t = t[valid]
    D = D[valid]

    if len(t) < 100:
        return np.nan

    logt = np.log(t[1:1000])
    logD = np.log(D[1:1000])

    slope = np.polyfit(logt, logD, 1)[0]
    return slope


def build_graph(net):
    G = nx.Graph()
    W = net['W']
    for i in range(len(W['from'])):
        G.add_edge(int(W['from'][i]), int(W['to'][i]))
    return G


# -----------------------------
# INPUT DATA
# -----------------------------

data = loadmat('datasets/Task1.mat')
U = np.array(data['dat']['u'][0][0][0:15000]) * 1e-11

# -----------------------------
# STORAGE
# -----------------------------

nodes_all = []
lyap_all = []
D_inf_all = []
D_var_all = []
alpha_all = []
nmse_all = []

example_D = {}   # store few curves

# -----------------------------
# MAIN LOOP
# -----------------------------

for nodes in range(4, 15):
    print("Nodes:", nodes)

    parameters = {
        'input_nodes': 1,
        'readout': 'LENGTHS',
        'rk_steps': 1000,
        'A': 4e-12,
        'L': [0e-6, 200e-6],
        'nodes': nodes,
        'x_lim': [0e-9, 199e-9],
        'y_lim': [0e-9, 199e-9],
        'basepair_range': [1e-6 / 0.63e-9, 200e-6 / 0.63e-9],
        'is_fixed_distance_bw_nodes': [False, 5e-9],
        'input_length': 50000,
        'time_step': 0.001,
        'show_steps': 1000,
        'b': 1.67e-7,
        'sequence': "".join([random.choice(["A","T","G","C"]) for _ in range(300)])
    }

    net_initial = network_system(init_ms_system, parameters)
    net_initial = change_to_random_connections(net_initial)

    # -------- NMSE --------
    seq = net_initial['seq']
    nmse = test_volterra_1(net_initial, seq, False)

    # -------- DAMAGE SPREADING --------
    net_clean = copy.deepcopy(net_initial)
    net_clean, sim_clean = simulate_dna_spring_reservoir(net_clean, U, parameters['b'])

    net_pert = copy.deepcopy(net_initial)
    net_pert['P']['states'][:, 0:2] += np.random.normal(0, 1e-12, net_pert['P']['states'][:, 0:2].shape)

    net_pert, sim_pert = simulate_dna_spring_reservoir(net_pert, U, parameters['b'])

    D = compute_damage(sim_clean, sim_pert)

    # -------- METRICS --------
    dt = parameters['time_step']
    lyap = compute_lyapunov(D, dt)
    D_inf = np.mean(D[-1000:])
    D_var = np.var(D[-5000:])
    alpha = compute_powerlaw_exponent(D)

    # store
    nodes_all.append(nodes)
    lyap_all.append(lyap)
    D_inf_all.append(D_inf)
    D_var_all.append(D_var)
    alpha_all.append(alpha)
    nmse_all.append(nmse)

    # save few curves
    if nodes in [4, 8, 14]:
        example_D[nodes] = D

    # -------- NETWORK PLOT --------
    G = build_graph(net_initial)
    plt.figure()
    nx.draw(G, node_size=50)
    plt.title(f"Network (nodes={nodes})")
    os.makedirs("plots/networks", exist_ok=True)
    plt.savefig(f"plots/networks/network_{nodes}.png")
    plt.close()


# -----------------------------
# SAVE DATA
# -----------------------------

df = pd.DataFrame({
    'nodes': nodes_all,
    'lyapunov': lyap_all,
    'D_inf': D_inf_all,
    'D_var': D_var_all,
    'alpha': alpha_all,
    'nmse': nmse_all
})

os.makedirs("results", exist_ok=True)
df.to_csv("results/full_analysis.csv", index=False)

# -----------------------------
# PLOTS
# -----------------------------

os.makedirs("plots", exist_ok=True)

# 1. Lyapunov vs nodes
plt.figure()
plt.plot(nodes_all, lyap_all, marker='o')
plt.axhline(0, linestyle='--')
plt.xlabel("Nodes")
plt.ylabel("Lyapunov")
plt.title("Lyapunov vs Network Size")
plt.savefig("plots/lyapunov_vs_nodes.png")

# 2. Damage steady
plt.figure()
plt.plot(nodes_all, D_inf_all, marker='o')
plt.xlabel("Nodes")
plt.ylabel("D_inf")
plt.title("Steady Damage")
plt.savefig("plots/Dinf_vs_nodes.png")

# 3. Variance peak
plt.figure()
plt.plot(nodes_all, D_var_all, marker='o')
plt.xlabel("Nodes")
plt.ylabel("Variance")
plt.title("Damage Variance (Criticality)")
plt.savefig("plots/variance.png")

# 4. NMSE vs Lyapunov
plt.figure()
plt.scatter(lyap_all, nmse_all)
plt.xlabel("Lyapunov")
plt.ylabel("NMSE")
plt.title("Performance vs Criticality")
plt.savefig("plots/nmse_vs_lyap.png")

# 5. Damage curves
plt.figure()
for k, v in example_D.items():
    plt.plot(v, label=f"{k} nodes")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Damage")
plt.title("Damage Evolution")
plt.savefig("plots/damage_curves.png")

# 6. Power-law check
plt.figure()
for k, v in example_D.items():
    plt.loglog(v, label=f"{k}")
plt.legend()
plt.title("Power-law scaling")
plt.savefig("plots/powerlaw.png")

print("ALL DONE ✅")