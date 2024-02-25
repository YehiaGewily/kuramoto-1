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

def plot_predicted_eeg(time_steps, avg_sine_phases, dt):
    plt.figure(figsize=(12, 6))
    time_steps= [i/dt for i in time_steps]
    avg_sine_phases= [j/dt for j in avg_sine_phases]
    plt.plot( time_steps, avg_sine_phases , label='Predicted EEG Signal')
    plt.xlabel('Time')
    plt.ylabel('Sine of Average Phase')
    plt.title('Predicted EEG Signal')
    plt.legend()
    plt.show()

