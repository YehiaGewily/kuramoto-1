import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.signal import hilbert, coherence
from mne.io import RawArray
from mne import create_info

# Function to compute Pearson correlation matrix
def compute_connectivity_matrix(data):
    correlation_matrix = np.corrcoef(data)
    return correlation_matrix

# Function to compute instantaneous phase using Hilbert transform
def compute_instantaneous_phase(data):
    analytic_signal = hilbert(data, axis=1)
    instantaneous_phase = np.angle(analytic_signal)
    return instantaneous_phase

# Function to compute Phase Locking Value (PLV) matrix
def compute_plv(instantaneous_phase):
    n_channels = instantaneous_phase.shape[0]
    plv_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            phase_diff = instantaneous_phase[i, :] - instantaneous_phase[j, :]
            plv_matrix[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
    return plv_matrix

# Function to compute Coherence matrix
def compute_coherence_matrix(data, sfreq):
    n_channels = data.shape[0]
    coh_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i, n_channels):
            f, Cxy = coherence(data[i], data[j], fs=sfreq)
            coh_matrix[i, j] = np.mean(Cxy)
            coh_matrix[j, i] = coh_matrix[i, j]  # Symmetric matrix
    return coh_matrix

# Function to compute Phase Lag Index (PLI) matrix
def compute_pli(instantaneous_phase):
    n_channels = instantaneous_phase.shape[0]
    pli_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            phase_diff = instantaneous_phase[i, :] - instantaneous_phase[j, :]
            pli_matrix[i, j] = np.abs(np.mean(np.sign(np.sin(phase_diff))))
    return pli_matrix

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

# Step 5: Compute connectivity matrices
# Pearson Correlation Matrix
correlation_matrix = compute_connectivity_matrix(eeg_data)

# Phase Locking Value (PLV) Matrix
instantaneous_phase = compute_instantaneous_phase(eeg_data)
plv_matrix = compute_plv(instantaneous_phase)

# Coherence Matrix
coherence_matrix = compute_coherence_matrix(eeg_data, sfreq)

# Phase Lag Index (PLI) Matrix
pli_matrix = compute_pli(instantaneous_phase)

# Step 6: Display and save the connectivity matrices

# Pearson Correlation Matrix
print("Pearson Correlation Matrix:")
correlation_df = pd.DataFrame(correlation_matrix, index=ch_names, columns=ch_names)
print(correlation_df)
correlation_df.to_csv('correlation_matrix.csv')

# PLV Matrix
print("Phase Locking Value (PLV) Matrix:")
plv_df = pd.DataFrame(plv_matrix, index=ch_names, columns=ch_names)
print(plv_df)
plv_df.to_csv('plv_matrix.csv')

# Coherence Matrix
print("Coherence Matrix:")
coherence_df = pd.DataFrame(coherence_matrix, index=ch_names, columns=ch_names)
print(coherence_df)
coherence_df.to_csv('coherence_matrix.csv')

# PLI Matrix
print("Phase Lag Index (PLI) Matrix:")
pli_df = pd.DataFrame(pli_matrix, index=ch_names, columns=ch_names)
print(pli_df)
pli_df.to_csv('pli_matrix.csv')

# Step 7: Visualize the connectivity matrices

def visualize_matrix(matrix, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(matrix, cmap='hot', interpolation='nearest')
    fig.colorbar(cax)
    ax.set_title(title)
    ax.set_xticks(np.arange(n_channels))
    ax.set_xticklabels(ch_names, rotation=90)
    ax.set_yticks(np.arange(n_channels))
    ax.set_yticklabels(ch_names)
    ax.set_xlabel('Channels')
    ax.set_ylabel('Channels')
    
    # Annotate the matrix with values
    for i in range(n_channels):
        for j in range(n_channels):
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='white' if matrix[i, j] > 0.5 else 'black')
    


# Pearson Correlation Matrix
visualize_matrix(correlation_matrix, 'Pearson Correlation Matrix')

# PLV Matrix
visualize_matrix(plv_matrix, 'Phase Locking Value (PLV) Matrix')

# Coherence Matrix
visualize_matrix(coherence_matrix, 'Coherence Matrix')

# PLI Matrix
visualize_matrix(pli_matrix, 'Phase Lag Index (PLI) Matrix')

# Optional: Create NetworkX graphs and visualize

def visualize_network(matrix, title):
    fig = plt.figure()
    G = nx.from_numpy_array(matrix)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', edge_color='k', font_weight='bold')
    plt.title(title)


# Pearson Correlation Matrix
visualize_network(correlation_matrix, 'Network Graph of Pearson Correlation Matrix')

# PLV Matrix
visualize_network(plv_matrix, 'Network Graph of PLV Matrix')

# Coherence Matrix
visualize_network(coherence_matrix, 'Network Graph of Coherence Matrix')

# PLI Matrix
visualize_network(pli_matrix, 'Network Graph of PLI Matrix')

plt.show()