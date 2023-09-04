import pandas as pd
import numpy as np
from numpy.linalg import svd
import itertools
import scipy
from scipy.linalg import eigh
import time
from sklearn.decomposition import TruncatedSVD

np.random.seed(0)


# todo: Remove future warnings


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


def rv(m1, m2):
    # Convert the datasets to numpy arrays for easier manipulation
    m1, m2 = np.array(m1), np.array(m2)
    # normed_scm1 and normed_scm2 are the "normed sums of cross products" of m1 and m2, respectively.
    normed_scm1 = m1.T @ m1
    normed_scm2 = m2.T @ m2
    # Calculate the RV coefficient using the formula.
    rv_index = np.sum(normed_scm1 * normed_scm2) / np.sqrt(
        np.sum(normed_scm1 * normed_scm2) * np.sum(normed_scm2 * normed_scm2))
    return rv_index


def pairwise_rv(dataset):
    dataset_names = list(dataset.keys())
    n = len(dataset)  # Number of datasets

    # For each combination, call the rv function with the 'weighted_table' of the corresponding datasets.
    RV = [rv(dataset[dataset_names[i]]['weighted_table'].values, dataset[dataset_names[j]]['weighted_table'].values)
          for i, j in itertools.combinations(range(n), 2)]

    m = np.ones((n, n))
    m[np.tril_indices(n, -1)] = RV
    m[np.triu_indices(n, 1)] = RV

    m = pd.DataFrame(m)
    # m.columns = m.index = list_of_names

    return m


def t_dudi(x):
    if not isinstance(x, dict) or 'eigenvalues' not in x.keys():
        raise ValueError("Dictionary of class 'dudi' expected")
    res = {'weighted_table': x['weighted_table'].transpose(), 'column_weight': x['row_weight'],
           'row_weight': x['column_weight'], 'eigenvalues': x['eigenvalues'], 'rank': x['rank'],
           'factor_numbers': x['factor_numbers'], 'component_scores': x['row_coordinates'],
           'row_coordinates': x['component_scores'], 'principal_coordinates': x['factor_scores'],
           'factor_scores': x['principal_coordinates'], 'dudi': 'transpo'}
    return res


def set_tab_names(x, value):
    if x['class'] != 'ktab':
        raise ValueError("to be used with 'ktab' object")

    ntab = len(x['blocks'])
    old = x['tab.names'][:ntab]

    if old and len(value) != len(old):
        raise ValueError("invalid tab.names length")

    value = list(map(str, value))

    if len(value) != len(set(value)):
        raise ValueError("duplicate tab.names are not allowed")

    x['tab.names'][:ntab] = value

    return x


def add_factor_to_ktab(ktab_dict):
    """
    This function adds additional factors to a k-table dictionary.
    It creates three new factors based on the blocks, row names, and column names.

    :param ktab_dict: A dictionary representing a k-table
    :return: The k-table dictionary with added factors
    """

    # Extract relevant information from the k-table dictionary
    block_sizes = list(ktab_dict['blocks'].values())
    num_rows = len(ktab_dict['row_weight'])
    num_blocks = len(ktab_dict['blocks'])
    row_names = ktab_dict['row.names']
    col_names = ktab_dict['col.names']
    block_names = ktab_dict['tab.names']

    # Construct the 'T' and 'L' factors
    T_factor = np.repeat(block_names, num_rows)  # Repeat each block name for each row
    L_factor = np.tile(row_names, num_blocks)  # Repeat the entire set of row names for each block
    TL_df = pd.DataFrame({'T': T_factor, 'L': L_factor})  # Combine into a DataFrame
    ktab_dict['TL'] = TL_df

    # todo: Tomorrow check this, but right now let's fix it like this, cause col_names is not right!

    T_factor = np.repeat(block_names, block_sizes)  # Repeat each block name for each row in that block
    C_factor = np.concatenate(col_names)

    TC_df = pd.DataFrame({'T': T_factor, 'C': C_factor})  # Combine into a DataFrame
    ktab_dict['TC'] = TC_df

    # Construct the 'T' and '4' factors
    T_factor = np.repeat(block_names, 4)  # Repeat each block name four times
    four_factor = np.tile(np.arange(1, 5), num_blocks)  # Repeat the sequence 1 to 4 for each block
    T4_df = pd.DataFrame({'T': T_factor, '4': four_factor})  # Combine into a DataFrame
    ktab_dict['T4'] = T4_df

    return ktab_dict


def compile_tables(objects, rownames=None, colnames=None):
    """
    The function compiles a list of tables (as dictionaries) with 'row_weight', 'column_weight', and 'weighted_table'
    attributes into a single output dictionary 'compiled_tables' to be passed to ktab_util_addfactor function.
    """

    # Check if all objects contain necessary attributes
    if not all(('row_weight' in item and 'column_weight' in item and 'weighted_table' in item) for item in objects):
        raise ValueError("list of objects with 'row_weight', 'column_weight', and 'weighted_table' attributes expected")

    num_blocks = len(objects)  # number of blocks
    compiled_tables = {}  # result dictionary to compile all objects
    row_weights = objects[0]['row_weight']  # get row weights from the first object
    column_weights = []  # placeholder for all column weights
    block_lengths = [item['weighted_table'].shape[1] for item in objects]  # lengths of all blocks

    for idx in range(num_blocks):
        if not np.array_equal(objects[idx]['row_weight'], row_weights):  # ensure all row weights are equal
            raise ValueError("Non equal row weights among arrays")

        compiled_tables[idx] = objects[idx]['weighted_table']  # add weighted table to result dictionary
        column_weights.extend(objects[idx]['column_weight'])  # add column weights to the list

    # get column names from all objects
    colnames_all_objects = [item['weighted_table'].columns.tolist() for item in objects]

    # check and set rownames
    if rownames is None:
        rownames = objects[0]['weighted_table'].index.tolist()
    elif len(rownames) != len(objects[0]['weighted_table'].index):
        raise ValueError("Non convenient rownames length")

    # check and set colnames
    if colnames is None:
        colnames = colnames_all_objects
    elif len(colnames) != len(colnames_all_objects):
        raise ValueError("Non convenient colnames length")

    # check for class attribute in objects and assign names accordingly
    tablenames = []
    for dictionary in objects:
        if 'class' in dictionary.keys():
            tablenames.append(dictionary['class'])
        else:
            tablenames.append(f"Ana{len(tablenames) + 1}")  # todo: This is the part that could be problematic

    # check and set tablenames
    if tablenames is None:
        tablenames = tablenames
    elif len(tablenames) != len(tablenames):
        raise ValueError("Non convenient tablenames length")

    # populate the rest of the result dictionary
    compiled_tables['blocks'] = dict(zip(tablenames, block_lengths))
    compiled_tables['row_weight'] = row_weights
    compiled_tables['column_weight'] = column_weights
    compiled_tables['class'] = 'ktab'
    compiled_tables['row.names'] = rownames
    compiled_tables['col.names'] = colnames
    compiled_tables['tab.names'] = tablenames

    compiled_tables = add_factor_to_ktab(compiled_tables)

    return compiled_tables


def scalewt(df, wt=None, center=True, scale=True):
    if wt is None:
        wt = np.repeat(1 / df.shape[0], df.shape[0])

    mean_df = None
    if center:
        mean_df = np.average(df, axis=0, weights=wt)
        df = df - mean_df

    var_df = None
    if scale:
        f = lambda x, w: np.sum(w * x ** 2) / np.sum(w)
        var_df = np.apply_along_axis(f, axis=0, arr=df, w=wt)
        temp = var_df < 1e-14
        if np.any(temp):
            import warnings
            warnings.warn("Variables with null variance not standardized.")
            var_df[temp] = 1
        var_df = np.sqrt(var_df)
        df = df / var_df

    attributes = {}
    if mean_df is not None:
        attributes['scaled:center'] = mean_df
    if var_df is not None:
        attributes['scaled:scale'] = var_df

    return df, attributes


def mcia(dataset, nf=2, nsc=True):
    """
    Performs multiple co-inertia analysis on a given set of datasets.

    Parameters:
    - dataset (list): List of datasets (pandas DataFrames) to analyze.
    - nf (int, default=2): Number of factors.
    - scan (bool, default=False): [unused in the provided function]
    - nsc (bool, default=True): Flag to decide if Non-Symmetric Correspondence Analysis is performed.
    - svd (bool, default=True): [unused in the provided function]

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
    if nsc:
        dataset = Array2Ade4(dataset, pos=True)

        # Perform Non-Symmetric Correspondence Analysis on each dataset
        nsca_results = {f'dataset_{i}': dudi_nsc(df, nf=nf) for i, df in enumerate(dataset)}  # Preprocessing

        # Store transformed results
        nsca_results_t = nsca_results

        # Perform t_dudi on weighted_table of each nsca_result
        for name, result in nsca_results.items():
            nsca_results_t[name] = t_dudi(result)

        # Calculate the pairwise RV coefficients
        RV = pairwise_rv(
            nsca_results)
        # RV coefficient is a way to define the information stored in two datasets, a value of 0 means
        # no relationship while 1 mean perfect agreement between the two datasets

        # Compile tables for analysis
        nsca_results_list = list(nsca_results.values())

        return nsca_results_list


def complete_dudi(dudi, nf1, nf2):
    """
    Augment the DUDI results with additional zero-filled columns for specified dimensions.

    Parameters:
    - dudi (dict): The original DUDI result containing various DataFrame structures.
    - nf1 (int): The start of the new range of axes.
    - nf2 (int): The end of the new range of axes.

    Returns:
    - dict: The updated DUDI result with additional zero-filled columns.
    """

    # A helper function to create a zero-filled DataFrame with custom columns
    def create_zero_df(rows, nf_start, nf_end):
        return pd.DataFrame(np.zeros((rows, nf_end - nf_start + 1)),
                            columns=[f'Axis{i}' for i in range(nf_start, nf_end + 1)])

    # Extend 'factor_scores' with zero-filled columns
    dudi['factor_scores'] = pd.concat([dudi['factor_scores'],
                                       create_zero_df(dudi['factor_scores'].shape[0], nf1, nf2)],
                                      axis=1)

    # Extend 'row_coordinates' with zero-filled columns
    dudi['row_coordinates'] = pd.concat([dudi['row_coordinates'],
                                         create_zero_df(dudi['row_coordinates'].shape[0], nf1, nf2)],
                                        axis=1)

    # Extend 'principal_coordinates' with zero-filled columns
    dudi['principal_coordinates'] = pd.concat([dudi['principal_coordinates'],
                                               create_zero_df(dudi['principal_coordinates'].shape[0], nf1, nf2)],
                                              axis=1)

    # Extend 'component_scores' with zero-filled columns
    dudi['component_scores'] = pd.concat([dudi['component_scores'],
                                          create_zero_df(dudi['component_scores'].shape[0], nf1, nf2)],
                                         axis=1)

    return dudi


def normalize_per_block(scorcol, number_of_blocks, block_indicator, veclev, tol=1e-7):
    """
    Normalize `scorcol` by block, based on the block indicators `indicablo`
    and the unique block levels `veclev`.
    This function is used to be sure that u_k is unitary

    Parameters:
    - scorcol: np.array, scores or values to be normalized.
    - nbloc: int, number of blocks.
    - indicablo: np.array, block indicators for each element in `scorcol`.
    - veclev: np.array, unique block levels.
    - tol: float, tolerance below which normalization is not performed.

    Returns:
    - np.array, the normalized `scorcol`.
    """
    for i in range(number_of_blocks):
        block_values = scorcol[block_indicator == veclev[i]]
        block_norm = np.sqrt(np.sum(block_values ** 2))

        if block_norm > tol:
            block_values /= block_norm

        scorcol[block_indicator == veclev[i]] = block_values

    return scorcol


def recalculate(tab, scorcol, nbloc, indicablo, veclev):
    """
    Adjust values in `tab` based on `scorcol` by block.
    This function is used to add the "deflation" method used by mcioa

    Parameters:
    - tab: pd.DataFrame, table to be adjusted.
    - scorcol: np.array, scores or values used for the adjustment.
    - nbloc: int, number of blocks.
    - indicablo: np.array, block indicators for columns in `tab`.
    - veclev: np.array, unique block levels.

    Returns:
    - pd.DataFrame, the adjusted table.
    """
    tab_np = tab.values
    indicablo_np = indicablo.values
    for k in range(nbloc):
        mask = indicablo_np == veclev[k]
        subtable = tab_np[:, mask]
        u_values = scorcol[mask]

        sum_values = (subtable * u_values).sum(axis=1)
        adjusted_subtable = subtable - np.outer(sum_values, u_values)

        tab_np[:, mask] = adjusted_subtable

    return pd.DataFrame(tab_np, columns=tab.columns)


def sepan(data, nf=20):
    """
    Compute successive eigenanalysis of partitioned data.

    Parameters:
    - data (dict): Contains the data, weights, block definitions, and other information.
    - nf (int): Number of factors to be computed.

    Returns:
    - dict: Results containing coordinates, scores, eigenvalues, etc.
    """

    # Ensure that the input data is of the expected type
    if data.get('class') != "ktab":
        raise ValueError("Expected object of class 'ktab'")

    lw = data['row_weight']
    cw = data['column_weight']
    blo = data['blocks']
    ntab = len(blo)
    j1 = 0
    j2 = list(blo.values())[0]

    auxi = as_dudi(data[0], col_w=cw[j1:j2], row_w=lw, nf=nf, class_type="sepan")

    if auxi['factor_numbers'] < nf:
        auxi = complete_dudi(auxi, auxi['factor_numbers'] + 1, nf)

    Eig = list(auxi['eigenvalues'])
    Co = auxi['principal_coordinates']
    Li = auxi['factor_scores']
    C1 = auxi['component_scores']
    L1 = auxi['row_coordinates']

    rank = [auxi['rank']]

    mapping = {
        'principal_coordinates': 'Co',
        'factor_scores': 'Li',
        'component_scores': 'C1',
        'row_coordinates': 'L1'
    }

    for df in [Li, L1, Co, C1]:
        df.index = [f'{index}.{j1}' for index in df.index]

    for i, block_key in enumerate(list(blo.keys())[1:], start=1):
        j1 = j2
        j2 = j2 + blo[block_key]
        tab = data[i]
        auxi = as_dudi(tab, cw[j1:j2], lw, nf=nf)

        # Append values to the respective lists
        Eig.extend(auxi['eigenvalues'].tolist())

        for key, short_name in mapping.items():
            auxi_df = auxi[key].copy()
            auxi_df.index = [f'X{idx + 1}.{block_key}' for idx in range(len(auxi_df))]
            if short_name == 'Co':
                Co = pd.concat([Co, auxi_df])
            elif short_name == 'Li':
                Li = pd.concat([Li, auxi_df])
            elif short_name == 'C1':
                C1 = pd.concat([C1, auxi_df])
            elif short_name == 'L1':
                L1 = pd.concat([L1, auxi_df])

        if auxi['factor_numbers'] < nf:
            auxi = complete_dudi(auxi, auxi['factor_numbers'] + 1, nf)

        rank.append(auxi['rank'])

    # Convert lists to desired data structures after the loop
    Eig = np.array(Eig)
    rank = np.array(rank)

    res = {
        'row_coordinates': L1,
        'component_scores': C1,
        'principal_coordinates': Co,
        'factor_scores': Li,
        'eigenvalues': Eig,
        'TL': data['TL'],
        'TC': data['TC'],
        'T4': data['T4'],
        'blocks': blo,
        'rank': rank,
        'tab_names': list(data.keys())[:ntab],
        'class': ["sepan", "list"]
    }

    return res


def ktab_util_names(x):
    """
    Generates utility names for ktab objects.

    Parameters:
    - x (dict): The ktab object with keys 'data', 'TL', 'TC', 'tab.names', and 'class'.

    Returns:
    - dict: A dictionary containing the row, column, and tab utility names. If 'kcoinertia' is in x['class'],
            an additional 'Trow' key-value pair is returned.
    """

    suffixes = [f"df{i + 1}" for i, t_val in enumerate(x['TL']['T'].unique())]
    row_names = []

    for idx, t_val in enumerate(x['TL']['T'].unique()):
        subset = x['TL'][x['TL']['T'] == t_val]
        row_names.extend([f"{l_val}.{suffixes[idx]}" for l_val in subset['L']])

    # Generate column names
    secondary_keys = x['col.names']

    unique_suffixes = [f"{str(i)}" for i in set(x['TC']['T'])]

    col_names = []
    for sublist, suffix in zip(secondary_keys, unique_suffixes):
        col_names.extend([f"{key}.{suffix}" for key in sublist])

    w = x['tab.names']
    tab_names_ktab = list()

    # Repeat the entire array 'w' 4 times
    # todo Check this with higher number of dataset
    for i in range(len(w)):
        for k in range(1, 5):
            tab_names_ktab.append(f"{w[i]}.{k}")

    # Check for 'kcoinertia' class
    if 'kcoinertia' not in x['class']:
        return {'row': row_names, 'col': col_names, 'tab': tab_names_ktab}

    # For 'kcoinertia' class, generate Trow names
    trow_names = [f"{i}.{j}" for i, count in zip(x['tab.names'], x['supblo']) for j in x['supX'][:count]]

    return {'row': row_names, 'col': col_names, 'tab': tab_names_ktab, 'Trow': trow_names}


def tab_names(x, value):
    """
    Assign or modify the 'tab.names' attribute of a 'ktab' object.

    Parameters:
    - x (dict): The ktab object which should have a 'blocks' key.
    - value (list): The list of tab names to assign.

    Returns:
    - dict: The updated ktab object with the new 'tab.names' attribute.

    Raises:
    - ValueError: If x is not a valid 'ktab' object, if the provided tab names length is invalid,
                  or if duplicate tab names are provided.
    """
    # Validate input
    if not hasattr(x, 'blocks'):
        raise ValueError("Function should be used with a 'ktab' object.")

    ntab = len(x['blocks'])
    old_names = x['tab.names'][:ntab] if 'tab.names' in x else None

    # Check the consistency of the new and old tab names
    if old_names is not None and len(value) != len(old_names):
        raise ValueError("Invalid tab.names length.")

    # Ensure no duplicate tab names
    value = [str(v) for v in value]
    if len(set(value)) != len(value):
        raise ValueError("Duplicate tab.names are not allowed.")

    # Assign new tab names
    x['tab.names'] = value[:ntab]

    return x


def mcoa(X, option=None, nf=3, data_projected = False):
    if option is None:
        option = ["inertia", "lambda1", "uniform", "internal"]
    if X.get('class') != "ktab":
        raise ValueError("Expected object of class 'ktab'")
    option = option[0]
    if option == "internal":
        if X.get('tabw') is None:
            print("Internal weights not found: uniform weights are used")
            option = "uniform"

    lw = X['row_weight']
    nlig = len(lw)
    cw = X['column_weight']
    ncol = len(cw)
    nbloc = len(X['blocks'])
    indicablo = X['TC']['T']
    veclev = sorted(list(set(X['TC']['T'])))
    #  todo: somehow this (the sorted argument) fixes the problem that sometimes the analysis order are swapped,
    #  causing wrong results. This is present somehow only in tests, but it seems like it's not a problem for a
    #  normal call of the functions. Will try to dive deeper into it after
    Xsepan = sepan(X, nf=4)  # This is used to calculate the component scores factor scores for each data

    rank_fac = list(np.repeat(range(1, nbloc + 1), Xsepan["rank"]))

    auxinames = ktab_util_names(X)
    sums = {}

    if option == "lambda1":
        tabw = [1 / Xsepan["eigenvalues"][rank_fac[i - 1]][0] for i in range(1, nbloc + 1)]
    elif option == "inertia":
        # Iterate over rank_fac and Xsepan['Eig'] simultaneously
        for rank, value in zip(rank_fac, Xsepan['eigenvalues']):
            # If rank is not in sums, initialize with value, otherwise accumulate
            sums[rank] = sums.get(rank, 0) + value

        # Create tabw by taking reciprocals of the accumulated sums
        tabw = [1 / sums[i] for i in sorted(sums.keys())]  # This is done to assign weights to each rank
    elif option == "uniform":
        tabw = [1] * nbloc
    elif option == "internal":
        tabw = X['tabw']
    else:
        raise ValueError("Unknown option")

    for i in range(nbloc):
        X[i] = [X[i] * np.sqrt(tabw[i])]  # We are weighting the datasets according to the calculated eigenvalues

    for k in range(nbloc):
        X[k] = pd.DataFrame(X[k][0])

    Xsepan = sepan(X, nf=4)  # Recalculate sepan with the updated X


    '''
    The call of two sepan functions, one with the non weighted dataset and the other with the weighted tables 
    is done so that the contributions of each datasets are balanced, so that the co-inertia structure better reflects
    shared patterns of the different datasets, not just patterns from the most variable dataset.
    Basically this is calculating \( X^† = [w^{\frac{1}{2}}_1 X_1, w^{\frac{1}{2}}_2 X_2, \dots, w^{\frac{1}{2}}_K X_K]\

    '''

    # Convert the first element of X to a DataFrame and assign it to tab
    tab = pd.DataFrame(X[0])

    if data_projected:
        return X[0]

    # Concatenate the remaining elements of X (from the 2nd to nbloc) to the columns of tab
    for i in range(1, nbloc):
        tab = pd.concat([tab, pd.DataFrame(X[i])], axis=1)

    '''
    This creates the merged table of K weighted datasets
    '''

    # Assign the names of the columns of tab from auxinames['col']
    tab.columns = auxinames['col']

    '''
    This is used to perform combined row and column weighting transformations to the data.
    The row weights, represented by \(D^{1/2}\), are multiplied with the data to adjust the influence 
    of each row (sample) in the final analysis. 
    This ensures that each row's contributions are scaled according to its importance or frequency.

    The column weights, represented by \(Q^{1/2}_k\), are multiplied with the data to adjust the influence 
    of each column (feature) in the corresponding data block \(X_k\). 
    This accounts for variable importance, scaling, or measurement units, ensuring that 
    each feature's contribution is balanced when combined with other datasets.

    These operations transform each data block \(X_k\) into \(\tilde{X}_k = w_k^{1/2} D^{1/2} X_k Q_k^{1/2}\),
    as per the This results in a weighted dataset that is used in the subsequent optimization problem.
    '''

    tab = tab.mul(np.sqrt(lw), axis=0)
    tab = tab.mul(np.sqrt(cw), axis=1)


    '''
    Initialization for upcoming calculations.
    compogene and uknorme are lists that will hold computation results, 
    while valsing might be for singular values, but we would need more context to be certain.
    '''
    compogene = []
    uknorme = []
    valsing = None

    '''
    Determine how many SVD iterations or components to compute,
    limiting the number of singular vectors/values to a maximum of 20.
    '''

    nfprovi = min(20, nlig, ncol)
    # Perform SVD computations
    for i in range(nfprovi):
        '''
        Compute the Singular Value Decomposition (SVD) of the weighted matrix tab.
        Mathematically, given a matrix A (tab), the SVD is represented as:
        \( A = U \Sigma V^T \)
        where U and V are orthogonal matrices and \(\Sigma\) is a diagonal matrix with singular values.
        '''

        truncated_svd = TruncatedSVD(n_components=nfprovi)
        u = truncated_svd.fit_transform(tab)
        s = truncated_svd.singular_values_
        vt = truncated_svd.components_
        u = u / s

        '''
        We extract the first column of U and the first row of V^T for a couple of key reasons:

        1. The first column of U (u[:, 0]) captures the most "effective" linear combination of rows 
           (e.g., genes in a gene expression dataset) in representing the original data.

        2. Similarly, the first row of V^T (vt[0, :]) captures the most "effective" linear combination of columns 
           (e.g., samples or cells in a gene expression dataset) in representing the original data.

        In essence, these vectors are the principal axes that maximize the variance in the data and are used 
        for the projections.
        '''
        # Extract the first column of u and normalize by the square root of lw (row_weights)
        normalized_u = u[:, 0] / np.sqrt(lw)
        compogene.append(normalized_u)  # Append to the list of compogene

        # Extract the first column of vt (v transposed in SVD), then normalize it
        normalized_v = normalize_per_block(vt[0, :], nbloc, indicablo, veclev)

        '''
        This deflationary step recalculates 'tab' to remove the influence of the previously computed 
        synthetic center 'normalized_v'. By doing this, we're setting the stage to find 
        the next orthogonal synthetic center in subsequent iterations. 
        The method ensures that the influence of each synthetic center is considered only once, 
        and in each subsequent iteration, 
        the "most influential" direction in the residual (or deflated) data is sought after.
        '''

        # Re-calculate tab
        tab = recalculate(tab, normalized_v, nbloc, indicablo, veclev)

        normalized_v /= np.sqrt(cw)
        uknorme.append(normalized_v)

        # Extract the first singular value
        singular_value = np.array([s[0]])

        valsing = np.concatenate([valsing, singular_value]) if valsing is not None else singular_value
    # Squaring the valsing to get pseudo eigenvalues
    pseudo_eigenvalues = valsing ** 2



    # Ensure nf is at least 2
    if nf <= 0:
        nf = 2

    acom = {'pseudo_eigenvalues': pseudo_eigenvalues}
    rank_fac = np.array(rank_fac)
    lambda_matrix = np.zeros((nbloc, nf))
    for i in range(1, nbloc + 1):
        mask = (rank_fac == i)

        # Filter out the eigenvalues using the mask
        w1 = Xsepan['eigenvalues'][mask]

        r0 = Xsepan['rank'][i - 1]
        if r0 > nf:
            r0 = nf

        # Assign values to the lambda_matrix
        lambda_matrix[i - 1, :r0] = w1[:r0]

    # Convert the matrix to a DataFrame and assign row and column names
    lambda_df = pd.DataFrame(lambda_matrix)
    lambda_df.index = Xsepan['tab_names']
    lambda_df.columns = [f'lam{i}' for i in range(1, nf + 1)]
    acom['lambda'] = lambda_df

    # Create a DataFrame for synthesized variables
    syn_var_df = pd.DataFrame(np.array(compogene).T[:, :nf])  # should contain the u_k
    syn_var_df.columns = [f'SynVar{i}' for i in range(1, nf + 1)]
    syn_var_df.index = X['row.names']
    acom['SynVar'] = syn_var_df

    # Create a DataFrame for axes
    axis_df = pd.DataFrame(np.array(uknorme).T[:, :nf])  # uknorme should be the v_k
    axis_df.columns = [f'Axis{i}' for i in range(1, nf + 1)]
    axis_df.index = auxinames['col']
    acom['axis'] = axis_df

    # Initialize matrices for further processing:

    # `w` is a matrix to store the transformed data for each data point across all blocks.
    w = np.zeros((nlig * nbloc, nf))

    # `covar` will store covariance-like values for each block.
    covar = np.zeros((nbloc, nf))
    i2 = 0
    current_index = 0  # Pointer for rows in `w`.
    # Start iterating over blocks.
    for k in range(nbloc):
        # Update the slice pointers.
        i2 = i2 + nlig

        # Create a mask for extracting data of the current block.
        mask = indicablo == veclev[k]

        '''
        Fetch the computed v_k values for the current block. 
        Mathematically:
        v_k = the kth vector from the right singular matrix corresponding to the kth block.
        After the weight multiplication we can see that this is equal to  QV
        '''
        vk = acom['axis'].reset_index(drop=True).loc[mask].values
        tab = np.array(X[k])

        cw_array = np.array(cw)
        vk *= cw_array[mask].reshape(-1, 1)

        '''
        Compute the product of the data matrix for the block and the weighted v_k values.
        This operation gives a projection of the data onto the direction of v_k.
        Mathematically:
        projection_k = X_k * QV
        '''
        projection = tab @ vk

        # Populate the `w` matrix with the computed values for the current block.
        rows_in_projection = projection.shape[0]
        w[current_index:current_index + rows_in_projection, :] = projection
        current_index += rows_in_projection

        '''
        Scale the projection values using the synthetic variables (u_k values).
        This operation ensures that the resulting values are in terms of how much they align 
        with the primary singular vectors. Then, it's scaled further using row weights.
        Mathematically:
        scaled_data = (projection_k * u) * D.
        '''
        scaled_data = (projection * acom['SynVar'].values) * lw.reshape(-1, 1)

        '''
        Sum the processed data for the current block. This step computes a cumulative 
        metric
        covar_k = sum(scaled_data)
        '''
        covar[k, :] = scaled_data.sum(axis=0)

    # Convert w to DataFrame with appropriate row names and column names
    w_df = pd.DataFrame(w, index=auxinames['row'])
    w_df.columns = [f"Axis{str(i + 1)}" for i in range(nf)]
    '''
    w is defined as X_k * VQ
    '''
    acom['Tli'] = w_df

    # Convert covar to DataFrame and square it, then store in acom
    covar_df = pd.DataFrame(covar)
    covar_df.index = X['tab.names']  # todo: check value
    covar_df.columns = [f"cov2{str(i + 1)}" for i in range(nf)]
    acom['cov2'] = covar_df ** 2

    # Initialize w and indices
    w = np.zeros((nlig * nbloc, nf))
    i2 = 0
    # Iterate over blocks to adjust w based on Tli and sqrt of lw
    for k in range(nbloc):
        i1 = i2 + 1
        i2 = i2 + nlig
        tab = acom['Tli'].iloc[i1 - 1:i2, :]
        lw_sqrt = np.sqrt(lw)
        squared_values = (tab.values * lw_sqrt.reshape(-1, 1)) ** 2
        column_sums_sqrt = np.sqrt(squared_values.sum(axis=0))
        tab = tab.divide(column_sums_sqrt)
        w[i1 - 1:i2, :] = tab.values

    # Create DataFrame for adjusted w and store it as Tl1 in acom
    w_df = pd.DataFrame(w, index=auxinames['row'])
    w_df.columns = [f"Axis{str(i + 1)}" for i in range(nf)]
    acom['Tl1'] = w_df  # a normalized and re-scaled version of Tli

    w = np.zeros((ncol, nf))
    i2 = 0
    # Iterate over blocks to update w based on SynVar and lw
    for k in range(nbloc):
        i1 = i2 + 1
        i2 = i2 + X[k].shape[1]
        urk = np.array(acom['SynVar'])
        tab = np.array(X[k])
        urk = urk * lw[:, np.newaxis]
        w[i1 - 1:i2, :] = tab.T.dot(urk)

    # Create DataFrame for w and store it as Tco in acom
    w_df = pd.DataFrame(w, index=auxinames['col'])
    w_df.columns = [f"SV{str(i + 1)}" for i in range(nf)]
    acom['Tco'] = w_df

    # Reset variables and initialize var.names
    var_names = []
    w = np.zeros((nbloc * 4, nf))
    i2 = 0
    block_indices = indicablo.reset_index(drop=True)

    # Iterate over blocks to update w and var.names based on axis and Xsepan
    for k in range(nbloc):
        i1 = i2 + 1
        i2 = i2 + 4

        bool_filter = block_indices == veclev[k]

        urk = acom['axis'].reset_index(drop=True)[bool_filter].values
        tab = Xsepan['component_scores'].reset_index(drop=True)[bool_filter].values
        bool_filter_array = np.array(bool_filter)
        filtered_cw = np.array([cw[i] for i, flag in enumerate(bool_filter_array) if flag]).reshape(-1, 1)
        urk = urk * filtered_cw
        tab = tab.T.dot(urk)
        tab = np.nan_to_num(
            tab)  # todo Check that this does not give error. The result for two datasets works, but could be a hard
        # coding. The concept is that this function produced NaN instead of 0.00000, must be checked.

        for i in range(min(nf, 4)):
            if tab[i, i] < 0:
                tab[i, :] = -tab[i, :]

        w[i1 - 1:i2, :] = tab
        var_names.extend([f"{Xsepan['tab_names'][k]}.a{str(i + 1)}" for i in range(4)])

    w_df = pd.DataFrame(w, index=auxinames['tab'])
    w_df.columns = [f"Axis{str(i + 1)}" for i in range(nf)]
    acom['Tax'] = w_df

    # Set additional properties of acom
    acom['nf'] = nf
    acom['TL'] = X['TL']
    acom['TC'] = X['TC']
    acom['T4'] = X['T4']
    acom['class'] = 'mcoa'

    return acom
