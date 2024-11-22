import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.decomposition import TruncatedSVD


def perform_ca_analysis(data_frame, nf=2):
    """
    Performs Correspondence Analysis (CA) on a provided DataFrame to analyze the relationships between rows and columns through dimensionality reduction.

    Parameters:
    - data_frame (pd.DataFrame): The DataFrame representing a two-dimensional contingency table.
    - nf (int, default=2): The number of dimensions (factors) to reduce the data to during the CA process.

    Returns:
    - np.array: The principal components extracted from the CA, representing the dataset in the reduced dimensionality space.

    Description:
    The function calculates the relative frequencies of the data, adjusts for the marginal totals, and applies a singular value decomposition (SVD) to derive the principal components. This process allows for the visualization and analysis of the patterns of association between the row and column categories in a lower-dimensional space.
    """
    X = data_frame.values
    row_totals = X.sum(axis=1)
    col_totals = X.sum(axis=0)
    grand_total = X.sum()

    P = X / grand_total

    Dr_inv_sqrt = np.diag(1 / np.sqrt(row_totals / grand_total))
    Dc_inv_sqrt = np.diag(1 / np.sqrt(col_totals / grand_total))

    r = row_totals / grand_total
    c = col_totals / grand_total
    rc_outer = np.outer(r, c)
    S = Dr_inv_sqrt @ (P - rc_outer) @ Dc_inv_sqrt
    transformed_df = pd.DataFrame(S, index=data_frame.index, columns=data_frame.columns)
    principal_components = decompose_data_to_principal_coords(transformed_df, np.ones(transformed_df.shape[1]), np.ones(transformed_df.shape[0]),
                                                              nf, transpose = False)
    return principal_components

def perform_pca_analysis(data_frame, nf=2):
    """
    Performs Principal Component Analysis (PCA) normalization on a given DataFrame.

    Parameters:
    - data_frame (pd.DataFrame): The DataFrame on which PCA is to be performed.
    - num_factors (int): The number of principal components to be extracted.

    Returns:
    - np.array: The principal components of the given DataFrame.

    Raises:
    - ValueError: If there are NA (Not Available) entries in the DataFrame.
    """
    row_weights = np.ones(data_frame.shape[0]) / data_frame.shape[0]
    column_weights = np.ones(data_frame.shape[1])

    if data_frame.isna().sum().sum() > 0:
        raise ValueError("The DataFrame contains NA entries.")

    calculate_column_mean = lambda col: np.sum(col * row_weights) / np.sum(row_weights)
    column_means = data_frame.apply(calculate_column_mean)
    centered_data_frame = data_frame.subtract(column_means, axis=1)

    calculate_column_norm = lambda col: np.sqrt(np.sum(col ** 2 * row_weights) / np.sum(row_weights))
    column_norms = centered_data_frame.apply(calculate_column_norm)
    column_norms[column_norms < 1e-08] = 1
    scaled_data_frame = centered_data_frame.div(column_norms, axis=1)

    principal_components = decompose_data_to_principal_coords(scaled_data_frame, column_weights, row_weights, nf, transpose=False)
    return principal_components

def perform_nsc_analysis(df, nf=2):
    """
    Performs Non-Symmetric Correspondence Analysis normalization on the data.

    Parameters:
    - df (pd.DataFrame or np.ndarray): The input data.
    - nf (int): The number of factors.

    Returns:
    - dict: A dictionary containing results of the analysis.
    """
    df = pd.DataFrame(df)
    col = df.shape[1]

    if (df.values < 0).any():
        raise ValueError("Negative entries in table")

    N = df.values.sum()
    if N == 0:
        raise ValueError("All entries are zero")

    row_w = df.sum(axis=1) / N
    col_w = df.sum(axis=0) / N
    df /= N

    transpose = False
    df = df.T.apply(lambda x: col_w if x.sum() == 0 else x / x.sum()).T
    df = df.subtract(col_w, axis=1)
    df *= col

    X = decompose_data_to_principal_coords(df, np.ones(col) / col, row_w, nf, transpose=transpose)
    X['N'] = N

    return X


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
            factor_scores.values @ component_scores)

        res['column_scores'] = pd.DataFrame(component_scores,
                                               columns=[f'CS{i + 1}' for i in range(nf)])
        res['row_scores'] = factor_scores
        res['row_scores'].columns = [f'Axis{i + 1}' for i in range(nf)]
        res['column_principal_coordinates'] = res['column_scores'].multiply(
            eigen_sqrt[::-1])
        res['row_principal_coordinates'] = res['row_scores'].div(eigen_sqrt[::-1])
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


def transpose_analysis_result(x):
    if not isinstance(x, dict) or 'eigenvalues' not in x.keys():
        raise ValueError("Dictionary of class 'dudi' expected")
    res = {'weighted_table': x['weighted_table'].transpose(), 'column_weight': x['row_weight'],
           'row_weight': x['column_weight'], 'eigenvalues': x['eigenvalues'], 'rank': x['rank'],
           'factor_numbers': x['factor_numbers'], 'column_scores': x['row_principal_coordinates'],
           'row_principal_coordinates': x['column_scores'], 'column_principal_coordinates': x['row_scores'],
           'row_scores': x['column_principal_coordinates'], 'dudi': 'transpo'}
    return res



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


def validate_data(dataset, pos=False, trans=False):
    """
    Ensures that the dataset meets certain criteria for further analyses. E.g: no NA

    Parameters:
    - dataset (list of pd.DataFrame): The input data.
    - pos (bool): If True, all negative values in the dataset are made positive.
    - trans (bool): If True, transpose the dataset.

    Returns:
    - list of pd.DataFrame: The processed data.
    """
    dataset = get_data(dataset)

    if isinstance(dataset, pd.DataFrame):
        dataset = [dataset]

    for i in range(len(dataset)):
        if dataset[i].isnull().values.any():
            raise ValueError("Array data must not contain NA values.")

        if pos and dataset[i].values.any() < 0:
            num = round(dataset[i].min().min()) - 1
            dataset[i] += abs(num)

        if trans:
            dataset[i] = dataset[i].T

    return dataset
