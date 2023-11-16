import pandas as pd
import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import TruncatedSVD


def perform_pca_analysis(data_frame, num_factors=2):
    """
    Performs Principal Component Analysis (PCA) on a given DataFrame.

    Parameters:
    - data_frame (pd.DataFrame): The DataFrame on which PCA is to be performed.
    - num_factors (int): The number of principal components to be extracted.

    Returns:
    - np.array: The principal components of the given DataFrame.

    Raises:
    - ValueError: If there are NA (Not Available) entries in the DataFrame.
    """
    total_values_sum = data_frame.values.sum()
    row_weights = np.ones(data_frame.shape[0]) / data_frame.shape[0]
    column_weights = np.ones(data_frame.shape[1])

    if data_frame.isna().sum().sum() > 0:
        raise ValueError("The DataFrame contains NA entries.")

    # Centering the DataFrame
    calculate_column_mean = lambda col: np.sum(col * row_weights) / np.sum(row_weights)
    column_means = data_frame.apply(calculate_column_mean)
    centered_data_frame = data_frame.subtract(column_means, axis=1)

    # Scaling the DataFrame
    calculate_column_norm = lambda col: np.sqrt(np.sum(col ** 2 * row_weights) / np.sum(row_weights))
    column_norms = centered_data_frame.apply(calculate_column_norm)
    column_norms[column_norms < 1e-08] = 1
    scaled_data_frame = centered_data_frame.div(column_norms, axis=1)

    principal_components = decompose_data_to_principal_coords(scaled_data_frame, column_weights, row_weights, num_factors, transpose=False)
    return principal_components


def perform_nsc_analysis(data_frame, num_factors=2):
    """
    Performs Non-Symmetric Correspondence Analysis (NSCA) on the given data.

    Parameters:
    - data_frame (pd.DataFrame or np.ndarray): The input data for analysis.
    - num_factors (int): The number of factors to extract.

    Returns:
    - dict: A dictionary containing the results of the NSCA.

    Raises:
    - ValueError: If there are negative entries in the data or if all frequencies are zero.
    """
    data_frame = pd.DataFrame(data_frame)
    num_columns = data_frame.shape[1]

    # Check for negative entries
    if (data_frame.values < 0).any():
        raise ValueError("The data contains negative entries.")

    total_sum = data_frame.values.sum()
    # Check if all frequencies are zero
    if total_sum == 0:
        raise ValueError("All frequencies in the data are zero.")

    # Calculating row and column weights
    row_weights = data_frame.sum(axis=1) / total_sum
    column_weights = data_frame.sum(axis=0) / total_sum
    normalized_data_frame = data_frame / total_sum

    # Adjusting the DataFrame
    normalized_data_frame = normalized_data_frame.T.apply(lambda x: column_weights if x.sum() == 0 else x / x.sum()).T
    normalized_data_frame = normalized_data_frame.subtract(column_weights, axis=1)
    normalized_data_frame *= num_columns

    principal_coords = decompose_data_to_principal_coords(normalized_data_frame, np.ones(num_columns) / num_columns, row_weights, num_factors, transpose=False)
    principal_coords['total_sum'] = total_sum

    return principal_coords


def decompose_data_to_principal_coords(df, col_w, row_w, nf=2, full=False, tol=1e-7, class_type=None, SVD=True, transpose=False):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Expected input is a pandas DataFrame.")

    lig, col = df.shape

    if len(col_w) != col:
        raise ValueError("Weights dimensions must match DataFrame dimensions.")
    if len(row_w) != lig:
        raise ValueError("Weights dimensions must match DataFrame dimensions.")
    if any(np.array(col_w) < 0):
        raise ValueError("Weights must be non-negative.")
    if any(np.array(row_w) < 0):
        raise ValueError("Weights must be non-negative.")

    if lig < col:
        transpose = True

    res = {'weighted_table': df.copy(), 'column_weight': col_w, 'row_weight': row_w}
    df_ori = df.copy()
    df = df.multiply(np.sqrt(row_w), axis=0)
    df = df.multiply(np.sqrt(col_w), axis=1)

    if SVD:
        if not transpose:
            X = df.values
            n_components = col
        else:
            X = df.T.values
            n_components = lig

        truncated = TruncatedSVD(n_components=n_components, tol=tol)
        truncated.fit(X)
        eig_values = truncated.singular_values_ ** 2
        eig_vectors = truncated.components_.T if not transpose else truncated.components_

    else:
        if not transpose:
            eigen_matrix = np.dot(df.T, df)
        else:
            eigen_matrix = np.dot(df, df.T)

        eig_values, eig_vectors = eigh(eigen_matrix)
        eig_values = eig_values[::-1]

    rank = sum((eig_values / eig_values[0]) > tol)
    nf = min(nf, rank)

    if full:
        nf = rank

    res['eigenvalues'] = eig_values[:rank]
    res['rank'] = rank
    res['factor_numbers'] = nf
    col_w = [1 if x == 0 else x for x in col_w]
    row_w = [1 if x == 0 else x for x in row_w]
    eigen_sqrt = np.sqrt(res['eigenvalues'][:nf])

    if not transpose:
        col_w_sqrt_rec = 1 / np.sqrt(col_w)
        component_scores = eig_vectors[:, -nf:] * col_w_sqrt_rec.reshape(-1, 1)
        factor_scores = df_ori.multiply(res['column_weight'], axis=1)
        factor_scores = pd.DataFrame(
            factor_scores.values @ component_scores)  # Matrix multiplication and conversion to DataFrame

        res['column_scores'] = pd.DataFrame(component_scores,
                                               columns=[f'CS{i + 1}' for i in range(nf)])  # principal axes (A)
        res['row_scores'] = factor_scores
        res['row_scores'].columns = [f'Axis{i + 1}' for i in range(nf)]  # row scores (L)
        res['column_principal_coordinates'] = res['column_scores'].multiply(
            eigen_sqrt[::-1])  # This is the column score (C)
        res['row_principal_coordinates'] = res['row_scores'].div(eigen_sqrt[::-1])  # This is the principal components (K)
    else:
        row_w_sqrt_rec = 1 / np.sqrt(row_w)
        row_coordinates = np.array(pd.DataFrame(eig_vectors.T).iloc[:, :nf] * row_w_sqrt_rec.reshape(-1, 1))
        factor_scores = df_ori.T.multiply(res['row_weight'], axis=1)
        factor_scores = pd.DataFrame(
            factor_scores.values @ row_coordinates)
        res['row_principal_coordinates'] = pd.DataFrame(row_coordinates, columns=[f'RS{i + 1}' for i in range(nf)])
        res['column_principal_coordinates'] = pd.DataFrame(factor_scores, columns=[f'Comp{i + 1}' for i in range(nf)])
        res['row_scores'] = res['row_principal_coordinates'].multiply(eigen_sqrt[::-1])
        res['column_scores'] = res['column_principal_coordinates'].div(eigen_sqrt[::-1])

    res['call'] = None
    if class_type is None:
        res['class'] = ['dudi']
    else:
        res['class'] = [class_type, "dudi"]
    return res


def transpose_analysis_result(analysis_result):
    """
    Transposes the analysis result from a 'dudi' class dictionary.

    Parameters:
    - analysis_result (dict): The analysis result dictionary, expected to be of class 'dudi'.

    Returns:
    - dict: A dictionary with transposed results.

    Raises:
    - ValueError: If the input is not a dictionary of class 'dudi'.
    """
    if not isinstance(analysis_result, dict) or 'eigenvalues' not in analysis_result:
        raise ValueError("Input must be a dictionary of class 'dudi'.")

    transposed_result = {
        'weighted_table': analysis_result['weighted_table'].transpose(),
        'column_weight': analysis_result['row_weight'],
        'row_weight': analysis_result['column_weight'],
        'eigenvalues': analysis_result['eigenvalues'],
        'rank': analysis_result['rank'],
        'factor_numbers': analysis_result['factor_numbers'],
        'column_scores': analysis_result['row_principal_coordinates'],
        'row_principal_coordinates': analysis_result['column_scores'],
        'column_principal_coordinates': analysis_result['row_scores'],
        'row_scores': analysis_result['column_principal_coordinates'],
        'dudi': 'transpo'
    }
    return transposed_result




def get_data(dataset):
    """
    Converts input data into a pandas DataFrame if possible.

    Parameters:
    - dataset (list, np.ndarray or pd.DataFrame): The input data.

    Returns:
    - pd.DataFrame: The data in DataFrame format.

    Raises:
    - ValueError: If dataset is a DataFrame containing non-numeric columns.
    """
    if isinstance(dataset, list):
        for i, data in enumerate(dataset):
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Item at index {i} is not a pandas DataFrame.")

    elif isinstance(dataset, np.ndarray) and np.isrealobj(dataset):
        dataset = pd.DataFrame(dataset)

    elif isinstance(dataset, pd.DataFrame):
        numeric_df = dataset.select_dtypes(include=[np.number])
        if dataset.shape[1] != numeric_df.shape[1]:
            raise ValueError("Array data was found to be a DataFrame but contains non-numeric columns.")

    else:
        raise ValueError("Input type not supported.")

    return dataset


def Array2Ade4(dataset, pos=False, trans=False):
    """
    Processes and transforms the dataset.

    Parameters:
    - dataset (list of pd.DataFrame): The input data.
    - pos (bool): If True, all negative values in the dataset are made positive.
    - trans (bool): If True, transpose the dataset.

    Returns:
    - list of pd.DataFrame: The processed data.
    """
    # Ensure the dataset items are DataFrames
    dataset = get_data(dataset)

    # Check if dataset is a single DataFrame, if so, convert it to a list of one dataframe
    if isinstance(dataset, pd.DataFrame):
        dataset = [dataset]

    for i in range(len(dataset)):
        # Check for NA values
        if dataset[i].isnull().values.any():
            print("Array data must not contain NA values.")
            exit(1)

        # Make negative values positive if 'pos' is True
        if pos and dataset[i].values.any() < 0:
            num = round(dataset[i].min().min()) - 1
            dataset[i] += abs(num)

        # Transpose the dataset if 'trans' is True
        if trans:
            dataset[i] = dataset[i].T

    return dataset