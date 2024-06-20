import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mne.io import RawArray
from mne import create_info

# Function to compute connectivity matrix using Pearson correlation
def compute_connectivity_matrix(data):
    correlation_matrix = np.corrcoef(data)
    return correlation_matrix

# Step 1: Load your EEG data from a text file
file_path = 'OpenBCI_GUI-v5-blinks-jawClench-alpha.txt'  # Replace with your actual file path
data = pd.read_csv(file_path, comment='%', header=None)

# Step 2: Extract EEG data and channel names
# The first column is the sample index, next 8 columns are EEG data
eeg_data = data.iloc[:, 1:9].values.T  # Transpose to have shape (n_channels, n_samples)
n_channels, n_samples = eeg_data.shape

# Step 3: Create MNE Info object
sfreq = 250  # Sample rate (Hz)
ch_names = [f'Channel{i+1}' for i in range(n_channels)]
info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

# Step 4: Create MNE Raw object
raw = RawArray(eeg_data, info)

# Step 5: Compute connectivity matrix (Pearson correlation in this example)
connectivity_matrix = compute_connectivity_matrix(eeg_data)

# Verify the diagonal elements of the connectivity matrix
print("Diagonal elements of the connectivity matrix:")
print(np.diag(connectivity_matrix))

# Display the actual connectivity matrix
print("Connectivity Matrix (Pearson Correlation):")
print(connectivity_matrix)

# Step 6: Save or Process Connectivity Matrix
np.savetxt('connectivity_matrix.csv', connectivity_matrix, delimiter=',')

# Optional: Visualize the Connectivity Matrix
plt.figure()
plt.imshow(connectivity_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Connectivity Matrix (Pearson Correlation)')
plt.xlabel('Channels')
plt.ylabel('Channels')


# Optional: Create a NetworkX graph and visualize
plt.figure()
G = nx.from_numpy_array(connectivity_matrix)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', edge_color='k', font_weight='bold')
plt.show()
