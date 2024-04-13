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
sns.set_context("notebook", font_scale=1.6)

# Interactions are represented as an adjacency matrix A, a 2D numpy ndarray.
# Instantiate a random graph and transform into an adjacency matrix
graph_nx = nx.erdos_renyi_graph(n=10, p=1)  # p=1 -> all-to-all connectivity
adj_mat = nx.to_numpy_array(graph_nx)

# Instantiate model with parameters
model = Kuramoto(coupling=2, dt=0.01, T=1000, n_nodes=len(adj_mat))

# Run simulation - output is time series for all nodes (node vs time)
activity = model.run(adj_mat=adj_mat)
act_mat= activity
time_steps, r_within_range, psi_within_range = model.calculate_psi_within_r_range(adj_mat=adj_mat, r_min=0.5, r_max=0.75)


# Plot all the time series
plot_activity(activity)
plot_phase_coherence(activity)

# Plot the predicted EEG signal
plot_predicted_eeg(time_steps, psi_within_range)

#Plot the heatmap of the oscillators
plot_phase_heatmap(activity)

# Select the phases of two oscillators to plot against each other
plot_phase_space_trajectory(activity)


plt.show()