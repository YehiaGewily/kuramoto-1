from kuramoto.kuramoto import Kuramoto
from kuramoto.plotting import plot_activity, plot_phase_coherence,  plot_predicted_eeg, plot_phase_heatmap, plot_phase_space_trajectory
import numpy as np
import matplotlib as plt
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import kuramoto
import pandas as pd


sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)

def main(n_runs=50):
    # Parameters
    n_nodes = 100  # Number of oscillators
    coupling_strength =2.5
    dt = 0.01
    T = 1000

    # Interactions are represented as an adjacency matrix A, a 2D numpy ndarray.
    graph_nx = nx.erdos_renyi_graph(n=n_nodes, p=1)  # p=1 -> all-to-all connectivity
    adj_mat = nx.to_numpy_array(graph_nx)

    # Initialize arrays to store metrics for each run
    order_params = np.zeros((n_runs, int(T/dt)))
    avg_sin_phases = np.zeros((n_runs, int(T/dt)))
    activities = []

    for i in range(n_runs):
        # Instantiate model with parameters
        model = Kuramoto(coupling=coupling_strength, dt=dt, T=T, n_nodes=n_nodes)

        # Run simulation - output is time series for all nodes (node vs time)
        activity = model.run(adj_mat=adj_mat)
        activities.append(activity)

        # Calculate order parameter and average sin(phase) for each time step
        for t in range(activity.shape[1]):
            angles = activity[:, t]
            order_params[i, t] = Kuramoto.phase_coherence(angles)
            avg_sin_phases[i, t] = np.mean(np.sin(angles))

    # Figure 1: Activity for each simulation
    fig, axes = plt.subplots(nrows=n_runs, figsize=(12, 2 * n_runs))
    for i, ax in enumerate(axes):
        ax.imshow(activities[i], aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'Activity for Simulation {i+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Oscillator')
    plt.tight_layout()
 

    # Figure 2: Order parameter for each simulation
    plt.figure(figsize=(12, 6))
    for i in range(n_runs):
        plt.plot(order_params[i, :], label=f'Run {i+1}')
    plt.title('Order Parameter over Time for Each Run')
    plt.xlabel('Time')
    plt.ylabel('Order Parameter')
    plt.legend()
 

    # Figure 3: Average sin(phase) for each simulation
    plt.figure(figsize=(12, 6))
    for i in range(n_runs):
        plt.plot(avg_sin_phases[i, :], label=f'Run {i+1}')
    plt.title('Average Sin(Phase) over Time for Each Run')
    plt.xlabel('Time')
    plt.ylabel('Average Sin(Phase)')
    plt.legend()
   

    # Figure 4: Average Order Parameter of all simulations
    plt.figure(figsize=(12, 6))
    mean_order_param = np.mean(order_params, axis=0)
    plt.plot(mean_order_param, label='Average Order Parameter')
    plt.title('Average Order Parameter of All Simulations')
    plt.xlabel('Time')
    plt.ylabel('Order Parameter')
    plt.legend()
   

    # Figure 5: Average of Average Sin(Phases) of all the simulations
    plt.figure(figsize=(12, 6))
    mean_avg_sin_phase = np.mean(avg_sin_phases, axis=0)
    plt.plot(mean_avg_sin_phase, label='Average of Average Sin(Phases)')
    plt.title('Average of Average Sin(Phases) of All Simulations')
    plt.xlabel('Time')
    plt.ylabel('Average Sin(Phase)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main(n_runs=10)  # You can specify the number of runs here
