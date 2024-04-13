import matplotlib.pyplot as plt
import numpy as np

from .kuramoto import Kuramoto


def plot_activity(activity):
    """
    Plot sin(angle) vs time for each oscillator time series.

    activity: 2D-np.ndarray
        Activity time series, node vs. time; ie output of Kuramoto.run()
    return:
        matplotlib axis for further customization
    """
    _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.sin(activity.T))
    ax.set_xlabel('Time', fontsize=25)
    ax.set_ylabel(r'$\sin(\theta)$', fontsize=25)
    return ax


def plot_phase_coherence(activity):
    """
    Plot order parameter phase_coherence vs time.

    activity: 2D-np.ndarray
        Activity time series, node vs. time; ie output of Kuramoto.run()
    return:
        matplotlib axis for further customization
    """
    _, ax = plt.subplots(figsize=(8, 3))
    ax.plot([Kuramoto.phase_coherence(vec) for vec in activity.T], 'o')
    ax.set_ylabel('Order parameter', fontsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylim((-0.01, 1))
    return ax

def plot_predicted_eeg(time_steps, psi_values):
    plt.figure(figsize=(12, 6))
    avg_sine_phases = np.sin(psi_values)
    plt.plot(time_steps, avg_sine_phases, label='Predicted EEG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Sine of Average Phase (ψ)')
    plt.title('Predicted EEG Signal')
    plt.legend()
 

def plot_phase_heatmap(activity, colormap='viridis', xlabel='Time Step', ylabel='Oscillator', 
                       title='Phase Heatmap of Kuramoto Oscillators'):
    plt.figure(figsize=(10, 6))
    plt.imshow(activity, aspect='auto', origin='lower', cmap=colormap,
               extent=[0, activity.shape[1], 0, activity.shape[0]])
    plt.colorbar(label='Phase (radians)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def plot_phase_space_trajectory(activity, oscillator_index_i=0, oscillator_index_j=1, 
                                point_style='r.', grid=True, xlabel='sin(phase of oscillator i)', 
                                ylabel='sin(phase of oscillator j)', title='Phase Space Trajectory of Two Coupled Oscillators'):
     oscillator_i = activity[3, :]  # First oscillator
     oscillator_j = activity[4, :]  # Second oscillator

     plt.figure(figsize=(8, 8))
     plt.plot(np.sin(oscillator_i), np.sin(oscillator_j), point_style, linestyle='none')
     plt.xlabel(xlabel)
     plt.ylabel(ylabel)
     plt.title(title)
     plt.grid(grid)
