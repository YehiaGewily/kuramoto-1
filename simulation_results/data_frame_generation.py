import pandas as pd
import os


# Or using forward slashes
file_path = 'D:/U the mind company/kuramoto/simulation_results/coupling_0.5_dt_0.01_T_1500_n_nodes_100/final_avg_sin_phases.csv'

data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to verify its contents
print(data.head())
