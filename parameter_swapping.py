import os
from kuramoto.kuramoto import Kuramoto
from kuramoto.plotting import plot_activity, plot_phase_coherence,  plot_predicted_eeg, plot_phase_heatmap, plot_phase_space_trajectory
import numpy as np
import matplotlib as plt
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import kuramoto
import pandas as pd

def main(n_runs=10):
    # Parameter ranges
    coupling_range = np.arange(0.5, 3.5, 0.5)
    dt_range = [0.01, 0.05, 0.1]
    T_range = [500, 1000, 1500]
    n_nodes_range = [10, 20, 50]

    # Directory for all results
    base_dir = "kuramoto_simulations"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # Iterate over all combinations of parameters
    for coupling in coupling_range:
        for dt in dt_range:
            for T in T_range:
                for n_nodes in n_nodes_range:
                    param_dir = f"coupling_{coupling}_dt_{dt}_T_{T}_n_nodes_{n_nodes}"
                    full_dir = os.path.join(base_dir, param_dir)
                    if not os.path.exists(full_dir):
                        os.mkdir(full_dir)

                    activities, order_params_list, avg_sin_phases_list = run_simulations(
                        n_nodes, coupling, dt, T, n_runs)

                    plot_aggregated_results(activities, order_params_list, avg_sin_phases_list, full_dir, 
                                            coupling, dt, T, n_nodes)

def run_simulations(n_nodes, coupling, dt, T, n_runs):
    activities = []
    order_params_list = []
    avg_sin_phases_list = []

    for _ in range(n_runs):
        graph_nx = nx.erdos_renyi_graph(n=n_nodes, p=0.8)
        adj_mat = nx.to_numpy_array(graph_nx)
        model = Kuramoto(coupling=coupling, dt=dt, T=T, n_nodes=n_nodes)
        activity = model.run(adj_mat=adj_mat)
        
        activities.append(activity)
        order_params = np.array([Kuramoto.phase_coherence(act) for act in activity.T])
        avg_sin_phases = np.mean(np.sin(activity), axis=0)
        
        order_params_list.append(order_params)
        avg_sin_phases_list.append(avg_sin_phases)

    return activities, order_params_list, avg_sin_phases_list

def plot_aggregated_results(activities, order_params_list, avg_sin_phases_list, dir_path, coupling, dt, T, n_nodes):
    n_runs = len(activities)

    # Plot aggregated order parameter and sin phases
    mean_order_params = np.mean(order_params_list, axis=0)
    mean_avg_sin_phases = np.mean(avg_sin_phases_list, axis=0)

    plt.figure(figsize=(12, 6))
    for i, order_params in enumerate(order_params_list):
        plt.plot(order_params, label=f'Run {i+1}')
    plt.title(f'Order Parameter Over Time (Avg: {np.mean(mean_order_params):.2f})\nCoupling={coupling}, dt={dt}, T={T}, Nodes={n_nodes}')
    plt.xlabel('Time')
    plt.ylabel('Order Parameter')
    plt.legend()
    plt.savefig(os.path.join(dir_path, 'order_parameters.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    for i, avg_sin_phases in enumerate(avg_sin_phases_list):
        plt.plot(avg_sin_phases, label=f'Run {i+1}')
    plt.title(f'Average Sin(Phase) Over Time (Avg: {np.mean(mean_avg_sin_phases):.2f})\nCoupling={coupling}, dt={dt}, T={T}, Nodes={n_nodes}')
    plt.xlabel('Time')
    plt.ylabel('Average Sin(Phase)')
    plt.legend()
    plt.savefig(os.path.join(dir_path, 'avg_sin_phases.png'))
    plt.close()

    # Aggregate figures
    plt.figure(figsize=(12, 6))
    plt.plot(mean_order_params, 'k-', label='Mean Order Parameter')
    plt.title(f'Mean Order Parameter of All Simulations\nCoupling={coupling}, dt={dt}, T={T}, Nodes={n_nodes}')
    plt.xlabel('Time')
    plt.ylabel('Order Parameter')
    plt.legend()
    plt.savefig(os.path.join(dir_path, 'mean_order_parameter.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(mean_avg_sin_phases, 'r-', label='Mean Avg Sin(Phases)')
    plt.title(f'Mean Avg Sin(Phases) of All Simulations\nCoupling={coupling}, dt={dt}, T={T}, Nodes={n_nodes}')
    plt.xlabel('Time')
    plt.ylabel('Average Sin(Phase)')
    plt.legend()
    plt.savefig(os.path.join(dir_path, 'mean_avg_sin_phase.png'))
    plt.close()

if __name__ == '__main__':
    main(n_runs=10)  # Specify number of runs per parameter set
