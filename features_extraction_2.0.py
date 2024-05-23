import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
from tabulate import tabulate
from colorama import Fore, Style

# Load Data
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Extract Time-Domain Features
def extract_time_domain_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    features = {
        "Mean": np.mean(signal),
        "Variance": np.var(signal),
        "Peak to Peak": np.ptp(signal),
        "RMS": rms,
        "Standard Deviation": np.std(signal),
        "Zero Crossing Rate": ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)
    }
    return features

# Extract Frequency-Domain Features
def extract_frequency_domain_features(signal, fs):
    freqs, psd = welch(signal, fs)
    spectral_entropy = entropy(psd)
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
    spectral_rolloff = freqs[np.cumsum(psd) >= 0.85 * np.sum(psd)][0] if len(freqs[np.cumsum(psd) >= 0.85 * np.sum(psd)]) > 0 else 0

    features = {
        "Delta Power": np.sum(psd[(freqs >= 0.5) & (freqs <= 4)]),
        "Theta Power": np.sum(psd[(freqs > 4) & (freqs <= 8)]),
        "Alpha Power": np.sum(psd[(freqs > 8) & (freqs <= 13)]),
        "Beta Power": np.sum(psd[(freqs > 13) & (freqs <= 30)]),
        "Gamma Power": np.sum(psd[(freqs > 30) & (freqs <= 100)]),
        "Spectral Entropy": spectral_entropy,
        "Spectral Centroid": spectral_centroid,
        "Spectral Rolloff": spectral_rolloff
    }
    return features

# Main Function
def main():
    filename = "D:/U the mind company/kuramoto/simulation_results/coupling_3.5_dt_0.01_T_1500_n_nodes_100/final_avg_sin_phases.csv"
    data = load_data(filename)
    signal = data.iloc[:, 1].values  # Adjust the index if necessary
    fs = 256  # Sampling frequency

    time_features = extract_time_domain_features(signal)
    freq_features = extract_frequency_domain_features(signal, fs)

    # Print time-domain features
    print(Fore.GREEN + "Time-Domain Features:")
    print(Style.RESET_ALL + tabulate(time_features.items(), headers=['Feature', 'Value'], tablefmt='grid'))

    # Print frequency-domain features
    print(Fore.CYAN + "Frequency-Domain Features:")
    print(Style.RESET_ALL + tabulate(freq_features.items(), headers=['Feature', 'Value'], tablefmt='grid'))

if __name__ == "__main__":
    main()
