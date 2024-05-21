import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from kuramoto.kuramoto import Kuramoto
import pandas as pd
from multiprocessing import Pool

def main(n_runs=10, num_processes=4):
    # Define simulation parameters
    coupling_range = np.arange(0.5, 4, 0.5)
    dt = 0.01
    T = 1500
    n_nodes = 100

    # Directory for results
    base_dir = "simulation_results"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # Set up multiprocessing pool
    with Pool(num_processes) as pool:
        for coupling in coupling_range:
            # Directory for this particular parameter configuration
            param_dir = f"coupling_{coupling}_dt_{dt}_T_{T}_n_nodes_{n_nodes}"
            param_path = os.path.join(base_dir, param_dir)
            os.makedirs(param_path, exist_ok=True)

            # Prepare parameters for all runs at current coupling
            params = [(n_nodes, coupling, dt, T, param_path) for _ in range(n_runs)]
            
            # Perform parallel simulations
            results = pool.map(run_simulation, params)
            
            # Compute the average of average sine phases for each coupling across all runs
            all_avg_sin_phases = [result for result in results]
            final_avg_sin_phases = np.mean(all_avg_sin_phases, axis=0)

            # Save and plot results for each coupling
            save_results(param_path, final_avg_sin_phases)

def run_simulation(params):
    n_nodes, coupling, dt, T, param_path = params
    graph_nx = nx.erdos_renyi_graph(n=n_nodes, p=0.8)
    adj_mat = nx.to_numpy_array(graph_nx)
    model = Kuramoto(coupling=coupling, dt=dt, T=T, n_nodes=n_nodes)
    activity = model.run(adj_mat=adj_mat)
    return np.mean(np.sin(activity), axis=0)

def save_results(param_path, final_avg_sin_phases):
    # Save the final averaged data to a CSV file
    csv_path = os.path.join(param_path, 'final_avg_sin_phases.csv')
    final_avg_df = pd.DataFrame(final_avg_sin_phases, columns=['Avg_Sin_Phases'])
    final_avg_df.to_csv(csv_path, index_label='Time')

    # Plot the final average sine phases
    plot_path = os.path.join(param_path, 'final_avg_sin_phases.png')
    plt.figure(figsize=(12, 6))
    plt.plot(final_avg_sin_phases, label='Final Avg Sin(Phases)')
    plt.title('Final Average of Average Sin(Phases) Across All Simulations')
    plt.xlabel('Time')
    plt.ylabel('Average Sin(Phase)')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

if __name__ == '__main__':
    main(n_runs=10, num_processes=4)  # Specify the number of parallel processes
