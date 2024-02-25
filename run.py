from kuramoto.kuramoto import Kuramoto
from kuramoto.plotting import plot_activity, plot_phase_coherence,  plot_predicted_eeg
import numpy as np
import matplotlib as plt
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import kuramoto

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.6)

# Interactions are represented as an adjacency matrix _A_, a 2D numpy ndarray.
# Instantiate a random graph and transform into an adjacency matrix
graph_nx = nx.erdos_renyi_graph(n=10, p=1)  # p=1 -> all-to-all connectivity
adj_mat = nx.to_numpy_array(graph_nx)

# Instantiate model with parameters
model = Kuramoto(coupling=1, dt=0.01, T=1000, n_nodes=len(adj_mat))
# Run simulation - output is time series for all nodes (node vs time)
activity = model.run(adj_mat=adj_mat)

# Plot all the time series
plot_activity(activity)
plot_phase_coherence(activity)
time_steps, avg_sine_phases = model.calculate_average_sine_phase(adj_mat=adj_mat)

# Plot the predicted EEG signal
plot_predicted_eeg(time_steps, avg_sine_phases, dt=0.01)
plt.show()