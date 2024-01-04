import pandas as pd
import numpy as np
#from .data_initialization import Array2Ade4
from .data_preprocessing import perform_nsc_analysis, transpose_analysis_result, Array2Ade4, perform_pca_analysis, perform_ca_analysis
#from .rv import pairwise_rv


def mcia(dataset, nf=10, analysis_type = "nsc"):
    """
    Performs multiple co-inertia analysis on a given set of datasets.

    Parameters:
    - dataset (list): List of datasets (pandas DataFrames) to analyze.
    - nf (int, default=2): Number of factors.
    - nsc (bool, default=True): Flag to decide if Non-Symmetric Correspondence Analysis is performed.

    Returns:
    - mciares (dict): Results containing mcoa and coa analyzes.
    """

    # Check if all items in dataset are pandas DataFrames
    for i, data in enumerate(dataset):
        if not isinstance(data, pd.DataFrame):
            print(f"Item at index {i} is not a pandas DataFrame.")
            return False

    # Ensure no feature in the datasets express in all observations
    for i, df in enumerate(dataset):
        minn = min(df.min())
        ind = df.apply(lambda x: np.all(x == minn), axis=1)
        if any(ind):
            print("Some features in the datasets do not express in all observation, remove them")
            exit(1)

    # Ensure the number of individuals is consistent across data frames
    total_columns = [df.shape[1] for df in dataset]
    if len(set(total_columns)) != 1:
        print("Non-equal number of individuals across data frames")
        exit(1)

    # Check for NA values in the datasets
    for i, data in enumerate(dataset):
        if data.isnull().values.any():
            print("There are NA values")
            exit(1)

    # Convert datasets to Ade4 format and perform Non-Symmetric Correspondence Analysis
    dataset = Array2Ade4(dataset, pos=True)

    # Perform Non-Symmetric Correspondence Analysis on each datase
    if analysis_type == "nsc":
        preprocessing_results = {f'dataset_{i}': perform_nsc_analysis(df, nf=nf) for i, df in enumerate(dataset)}
    elif analysis_type == "pca":
        preprocessing_results = {f'dataset_{i}': perform_pca_analysis(df, nf=nf) for i, df in enumerate(dataset)}
    elif analysis_type == "ca":
        preprocessing_results = {f'dataset_{i}': perform_ca_analysis(df, nf=nf) for i, df in enumerate(dataset)}
    # Store transformed results
    nsca_results_t = preprocessing_results
    nsca_results = preprocessing_results
    # Perform t_dudi on weighted_table of each nsca_result
    for name, result in nsca_results.items():
        nsca_results_t[name] = transpose_analysis_result(result)

    # Calculate the pairwise RV coefficients
    #RV = pairwise_rv(
    #    nsca_results)
    # RV coefficient is a way to define the information stored in two datasets, a value of 0 means
    # no relationship while 1 mean perfect agreement between the two datasets

    # Compile tables for analysis
    nsca_results_list = list(nsca_results.values())

    return nsca_results_list
