import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import EfficientFCParameters

def load_and_prepare_data(file_path):
    # Load the data from the specified CSV file
    data = pd.read_csv(file_path)
    
    # Prepare the data for tsfresh:
    # Adding a constant ID since tsfresh requires an ID column to differentiate multiple time series
    data['id'] = 1  
    # Ensure there is a 'time' column to sort the data in temporal order
    data['time'] = range(len(data))  
    return data

def extract_and_save_features(data, output_file_path):
    # Define the settings for a more efficient feature extraction
    settings = EfficientFCParameters()

    # Extract features using the specified settings and disable parallel processing to simplify troubleshooting
    extracted_features = extract_features(data, column_id='id', column_sort='time', default_fc_parameters=settings, n_jobs=1)

    # Impute any missing values from the feature extraction
    imputed_features = impute(extracted_features)

    # Save the extracted (and imputed) features to a new CSV file
    imputed_features.to_csv(output_file_path)
    
    return imputed_features

def main():
    # Define the file paths for the input data and output file
    input_file_path = 'D:/U the mind company/kuramoto/simulation_results/coupling_0.5_dt_0.01_T_1500_n_nodes_100/final_avg_sin_phases.csv'
    output_file_path = 'D:/U the mind company/kuramoto/simulation_results/coupling_0.5_dt_0.01_T_1500_n_nodes_100/extracted_features.csv'
    
    # Load and prepare the data
    data = load_and_prepare_data(input_file_path)
    
    # Extract features and save them
    features = extract_and_save_features(data, output_file_path)
    
    # Print some of the extracted features to verify the output
    print(features.head())

if __name__ == '__main__':
    main()
