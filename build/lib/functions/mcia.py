import pandas as pd
import numpy as np
from .data_preprocessing import perform_nsc_analysis, transpose_analysis_result, checks, perform_pca_analysis, perform_ca_analysis


def mcia(dataset, nf=10, analysis_type = "nsc"):
    """
    Performs pre-processing on a given set of datasets.

    Parameters:
    - dataset (list): List of datasets (pandas DataFrames) to analyze.
    - nf (int, default=2): Number of factors.
    - nsc (bool, default=True): Flag to decide if Non-Symmetric Correspondence Analysis is performed.

    Returns:
    - mciares (dict): Results containing mcoa and coa analyzes.
    """

    for i, data in enumerate(dataset):
        if not isinstance(data, pd.DataFrame):
            print(f"Item at index {i} is not a pandas DataFrame.")
            return False

    for i, df in enumerate(dataset):
        minn = min(df.min())
        ind = df.apply(lambda x: np.all(x == minn), axis=1)
        if any(ind):
            print("Some features in the datasets do not express in all observation, remove them")
            exit(1)

    total_columns = [df.shape[1] for df in dataset]
    if len(set(total_columns)) != 1:
        print("Non-equal number of individuals across data frames")
        exit(1)

    for i, data in enumerate(dataset):
        if data.isnull().values.any():
            print("There are NA values")
            exit(1)

    dataset = checks(dataset, pos=True)

    analysis_functions = {
        "nsc": perform_nsc_analysis,
        "pca": perform_pca_analysis,
        "ca": perform_ca_analysis
    }

    if analysis_type not in analysis_functions:
        raise ValueError(f"Analysis type not found: {analysis_type}")

    preprocessing_results = {f'dataset_{i}': analysis_functions[analysis_type](df, nf=nf) for i, df in enumerate(dataset)}

    nsca_results_t = preprocessing_results
    nsca_results = preprocessing_results

    for name, result in nsca_results.items():
        nsca_results_t[name] = transpose_analysis_result(result)

    nsca_results_list = list(nsca_results.values())

    return nsca_results_list
