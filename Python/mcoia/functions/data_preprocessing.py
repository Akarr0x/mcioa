import pandas as pd
import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import TruncatedSVD


def dudi_nsc(df, nf=2):
    """
    Performs Non-Symmetric Correspondence Analysis on the data.

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
        raise ValueError("All frequencies are zero")

    row_w = df.sum(axis=1) / N
    col_w = df.sum(axis=0) / N
    df /= N

    # Transpose if more rows than columns
    transpose = False
    #if df.shape[1] > df.shape[0]:
    #    transpose = True
    #    df = df.T
    #    col, row_w, col_w = df.shape[1], col_w, row_w  # Swap row and column weights

    # Normalize and center data
    df = df.T.apply(lambda x: col_w if x.sum() == 0 else x / x.sum()).T
    df = df.subtract(col_w, axis=1)
    df *= col

    X = as_dudi(df, np.ones(col) / col, row_w, nf, transpose=transpose)
    X['N'] = N

    return X


def as_dudi(df, col_w, row_w, nf=2, full=False, tol=1e-7, class_type=None, SVD=True, transpose=False):
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

        res['component_scores'] = pd.DataFrame(component_scores,
                                               columns=[f'CS{i + 1}' for i in range(nf)])  # principal axes (A)
        res['factor_scores'] = factor_scores
        res['factor_scores'].columns = [f'Axis{i + 1}' for i in range(nf)]  # row scores (L)
        res['principal_coordinates'] = res['component_scores'].multiply(
            eigen_sqrt[::-1])  # This is the column score (C)
        res['row_coordinates'] = res['factor_scores'].div(eigen_sqrt[::-1])  # This is the principal components (K)
    else:
        row_w_sqrt_rec = 1 / np.sqrt(row_w)
        row_coordinates = np.array(pd.DataFrame(eig_vectors.T).iloc[:, :nf] * row_w_sqrt_rec.reshape(-1, 1))
        factor_scores = df_ori.T.multiply(res['row_weight'], axis=1)
        factor_scores = pd.DataFrame(
            factor_scores.values @ row_coordinates)
        res['row_coordinates'] = pd.DataFrame(row_coordinates, columns=[f'RS{i + 1}' for i in range(nf)])
        res['principal_coordinates'] = pd.DataFrame(factor_scores, columns=[f'Comp{i + 1}' for i in range(nf)])
        res['factor_scores'] = res['row_coordinates'].multiply(eigen_sqrt[::-1])
        res['component_scores'] = res['principal_coordinates'].div(eigen_sqrt[::-1])

    res['call'] = None
    if class_type is None:
        res['class'] = ['dudi']
    else:
        res['class'] = [class_type, "dudi"]
    return res


def t_dudi(x):
    if not isinstance(x, dict) or 'eigenvalues' not in x.keys():
        raise ValueError("Dictionary of class 'dudi' expected")
    res = {'weighted_table': x['weighted_table'].transpose(), 'column_weight': x['row_weight'],
           'row_weight': x['column_weight'], 'eigenvalues': x['eigenvalues'], 'rank': x['rank'],
           'factor_numbers': x['factor_numbers'], 'component_scores': x['row_coordinates'],
           'row_coordinates': x['component_scores'], 'principal_coordinates': x['factor_scores'],
           'factor_scores': x['principal_coordinates'], 'dudi': 'transpo'}
    return res