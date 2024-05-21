import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import fft
from scipy.stats import entropy

# Load Data
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Extract Time-Domain Features
def extract_time_domain_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    features = {
        "mean": np.mean(signal),
        "variance": np.var(signal),
        "peak_to_peak": np.ptp(signal),
        "rms": rms,
        "std_dev": np.std(signal),
        "zero_crossing_rate": ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)
    }
    return features

# Extract Frequency-Domain Features
def extract_frequency_domain_features(signal, fs):
    freqs, psd = welch(signal, fs)
    spectral_entropy = entropy(psd)
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
    spectral_rolloff = freqs[np.cumsum(psd) >= 0.85 * np.sum(psd)][0] if len(freqs[np.cumsum(psd) >= 0.85 * np.sum(psd)]) > 0 else 0

    features = {
        "delta_power": np.sum(psd[(freqs >= 0.5) & (freqs <= 4)]),
        "theta_power": np.sum(psd[(freqs > 4) & (freqs <= 8)]),
        "alpha_power": np.sum(psd[(freqs > 8) & (freqs <= 13)]),
        "beta_power": np.sum(psd[(freqs > 13) & (freqs <= 30)]),
        "gamma_power": np.sum(psd[(freqs > 30) & (freqs <= 100)]),
        "spectral_entropy": spectral_entropy,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff
    }
    return features

# Perform FFT and Plot
def plot_fft(signal, fs):
    n = len(signal)
    t = 1/fs
    yf = fft(signal)
    xf = np.linspace(0.0, 1.0/(2.0*t), n//2)
    plt.figure(figsize=(12, 6))
    plt.plot(xf, 2.0/n * np.abs(yf[:n//2]))
    plt.grid()
    plt.title('FFT of the Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()

# Main Function
def main():
    filename = "D:/U the mind company/kuramoto/simulation_results/coupling_0.5_dt_0.01_T_1500_n_nodes_100/final_avg_sin_phases.csv"
    data = load_data(filename)
    signal = data.iloc[:, 1].values  # Adjust the index if necessary
    fs = 256  # Sampling frequency

    time_features = extract_time_domain_features(signal)
    freq_features = extract_frequency_domain_features(signal, fs)

    print("Time-Domain Features:", time_features)
    print("Frequency-Domain Features:", freq_features)

    # Plot FFT
    plot_fft(signal, fs)

if __name__ == "__main__":
    main()
