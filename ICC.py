import pandas as pd
import numpy as np
import networkx as nx
from kuramoto.kuramoto import Kuramoto
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

# Setup visualization style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.6)

# Parameters
n_runs = 10  # Specify the number of simulations
T = 1000  # Total simulation time
dt = 0.01  # Time step

# Prepare DataFrame to store results
# Assuming T/dt time steps, create an index representing each time step
time_steps = np.arange(0, T, dt)
results = pd.DataFrame(index=time_steps)

# Simulation
for run in range(n_runs):
    # Generate a random graph and adjacency matrix for connectivity
    graph_nx = nx.erdos_renyi_graph(n=10, p=1)  # Full connectivity with p=1
    adj_mat = nx.to_numpy_array(graph_nx)
    
    # Initialize the Kuramoto model
    model = Kuramoto(coupling=2, dt=dt, T=T, n_nodes=len(adj_mat))
    
    # Run the model simulation
    activity = model.run(adj_mat=adj_mat)
    
    # Calculate the order parameter for each time step
    order_params = [Kuramoto.phase_coherence(activity[:, t]) for t in range(activity.shape[1])]
    
    # Store the order parameters for this run
    results[f'Run_{run+1}'] = order_params

print(results)

if results.isnull().values.any():
    print("Warning: Missing data detected. Check simulation outputs.")

# Ensure the data matches in length (debugging)
assert all(len(results[col]) == len(time_steps) for col in results.columns), "Data length mismatch."

# Reshape the dataframe for ICC calculation
icc_data = results.melt(var_name='Run', value_name='OrderParameter')
icc_data['Time'] = icc_data.index

print(icc_data)

# Include the testing code here
# Let's take a small subset for testing
test_data = icc_data.head(50)  # Adjust this number as needed for a meaningful subset

# Attempt ICC calculation on this smaller subset
try:
    test_icc = pg.intraclass_corr(data=test_data, targets='Time', raters='Run', ratings='OrderParameter', nan_policy='omit').round(3)
    print("Test ICC calculation on a subset of the data:", test_icc)
except Exception as e:
    print("Error encountered with test dataset:", str(e))


# Calculate ICC
icc = pg.intraclass_corr(data=icc_data, targets='Time', raters='Run', ratings='OrderParameter', nan_policy='omit').round(3)
print(icc)

# Note: You might need to adjust the logic for collecting and structuring the data based on
# the actual output format of your Kuramoto model's run method and the order parameter calculation.
