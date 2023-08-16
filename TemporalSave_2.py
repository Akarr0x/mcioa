import pandas as pd
import numpy as np
from numpy.linalg import svd
import itertools
import matplotlib as plt
from scipy.linalg import eigh


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
            num = round(dataset[i].values.min()) - 1
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
    if df.shape[1] > df.shape[0]:
        transpose = True
        df = df.T
        col, row_w, col_w = df.shape[1], col_w, row_w  # Swap row and column weights

    # Normalize and center data
    df = df.T.apply(lambda x: col_w if x.sum() == 0 else x / x.sum()).T
    df = df.subtract(col_w, axis=1)
    df *= col

    X = as_dudi(df, np.ones(col) / col, row_w, nf)
    X['N'] = N

    return X


def as_dudi(df, col_w, row_w, nf=2, scannf=False, full=False, tol=1e-7, type=None):
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

    transpose = False
    if lig < col:
        transpose = True

    res = {'weighted_table': df.copy(), 'column_weight': col_w, 'row_weight': row_w}
    df_ori = df.copy()
    df = df.multiply(np.sqrt(row_w), axis=0)
    df = df.multiply(np.sqrt(col_w), axis=1)

    if not transpose:
        eigen_matrix = np.dot(df.T, df)
    else:
        eigen_matrix = np.dot(df, df.T)

    eig_values, eig_vectors = eigh(eigen_matrix)  # TODO check if SVD is faster
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
    dval = np.sqrt(res['eigenvalues'][:nf])

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
        res['principal_coordinates'] = res['component_scores'].multiply(dval[::-1])  # This is the column score (C)
        res['row_coordinates'] = res['factor_scores'].div(dval[::-1])  # This is the principal components (K)
    else:
        row_w_sqrt_rec = 1 / np.sqrt(row_w)
        row_coordinates = eig_vectors[:, -nf:] * row_w_sqrt_rec.to_numpy().reshape(-1, 1)
        principal_coordinates = (df.T.multiply(res['row_weight'], axis='columns') @ row_coordinates).T
        res['row_coordinates'] = pd.DataFrame(row_coordinates, columns=[f'RS{i + 1}' for i in range(nf)])
        res['principal_coordinates'] = pd.DataFrame(principal_coordinates, columns=[f'Comp{i + 1}' for i in range(nf)])
        res['factor_scores'] = res['row_coordinates'].multiply(dval[::-1])
        res['component_scores'] = res['principal_coordinates'].div(dval[::-1])

    res['call'] = None
    if type is None:
        res['class'] = ['dudi']
    else:
        res['class'] = [type, "dudi"]
    return res


def rv(m1, m2):
    # Convert the datasets to numpy arrays for easier manipulation
    m1, m2 = np.array(m1), np.array(m2)
    # nscm1 and nscm2 are the "normed sums of cross products" of m1 and m2, respectively.
    nscm1 = m1.T @ m1
    nscm2 = m2.T @ m2
    # Calculate the RV coefficient using the formula.
    rv = np.sum(nscm1 * nscm2) / np.sqrt(np.sum(nscm1 * nscm1) * np.sum(nscm2 * nscm2))
    return rv


def pairwise_rv(dataset):
    dataset_names = list(dataset.keys())
    n = len(dataset)  # Number of datasets

    # For each combination, call the rv function with the 'weighted_table' of the corresponding datasets.
    # Now we just index the dataset list directly, without using keys.
    RV = [rv(dataset[dataset_names[i]]['weighted_table'].values, dataset[dataset_names[j]]['weighted_table'].values)
          for i, j in itertools.combinations(range(n), 2)]

    m = np.ones((n, n))
    m[np.tril_indices(n, -1)] = RV
    m[np.triu_indices(n, 1)] = RV

    m = pd.DataFrame(m)
    # If you have names/labels for each dictionary in the dataset list, you can assign them here.
    # Otherwise, this step can be omitted.
    # m.columns = m.index = list_of_names

    return m


def t_dudi(x):
    if not isinstance(x, dict) or 'eigenvalues' not in x.keys():
        raise ValueError("Dictionary of class 'dudi' expected")
    res = {}
    res['weighted_table'] = x['weighted_table'].transpose()
    res['column_weight'] = x['row_weight']
    res['row_weight'] = x['column_weight']
    res['eigenvalues'] = x['eigenvalues']
    res['rank'] = x['rank']
    res['factor_numbers'] = x['factor_numbers']
    res['component_scores'] = x['row_coordinates']
    res['row_coordinates'] = x['component_scores']
    res['principal_coordinates'] = x['factor_scores']
    res['factor_scores'] = x['principal_coordinates']
    res['dudi'] = 'transpo'
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

    # Construct a sequence for each block size
    sequence = None
    for block_size in block_sizes:
        if sequence is None:
            sequence = np.arange(1, block_size + 1)
        else:
            sequence = np.append(sequence, np.arange(1, block_size + 1))

    # Construct the 'T' and 'C' factors
    T_factor = np.repeat(block_names, num_rows)  # Repeat each block name for each row
    C_factor = np.tile(col_names[0], num_blocks)  # Repeat the first column name for each block
    TC_df = pd.DataFrame({'T': T_factor, 'C': C_factor})  # Combine into a DataFrame
    ktab_dict['TC'] = TC_df

    # Construct the 'T' and '4' factors
    T_factor = np.repeat(block_names, 4)  # Repeat each block name four times
    four_factor = np.tile(np.arange(1, 5), num_blocks)  # Repeat the sequence 1 to 4 for each block
    T4_df = pd.DataFrame({'T': T_factor, '4': four_factor})  # Combine into a DataFrame
    ktab_dict['T4'] = T4_df

    return ktab_dict


def compile_tables(objects, rownames=None, colnames=None, tablenames=None):
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
            tablenames.append(f"Ana{len(tablenames) + 1}")

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

    # Printing the final form of 'compiled_tables' dictionary before it's passed to ktab_util_addfactor function.
    print(compiled_tables)

    # Passing the 'compiled_tables' to 'ktab_util_addfactor' function
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
            # In Python, we generally raise a warning using the warnings module
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


def mcia(dataset, nf=2, scan=False, nsc=True, svd=True):
    """
    Performs multiple co-inertia analysis on a given set of datasets.

    Parameters:
    - dataset (list): List of datasets (pandas DataFrames) to analyze.
    - nf (int, default=2): Number of factors.
    - scan (bool, default=False): [unused in the provided function]
    - nsc (bool, default=True): Flag to decide if Non-Symmetric Correspondence Analysis is performed.
    - svd (bool, default=True): [unused in the provided function]

    Returns:
    - mciares (dict): Results containing mcoa and coa analyses.
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
        nsca_results = {f'dataset_{i}': dudi_nsc(df, nf=nf) for i, df in enumerate(dataset)}

        # Store transformed results
        nsca_results_t = nsca_results

        # Perform t_dudi on weighted_table of each nsca_result
        for name, result in nsca_results.items():
            nsca_results_t[name] = t_dudi(result)

        # Calculate the pairwise RV coefficients
        RV = pairwise_rv(nsca_results)

        # Compile tables for analysis
        nsca_results_list = list(nsca_results.values())
        ktcoa = compile_tables(nsca_results_list)

        # Perform MCoA
        mcoin = mcoa(X=ktcoa, nf=nf, tol=1e-07)

        # Scale the results
        tab, attributes = scalewt(mcoin['Tco'], ktcoa['cw'], center=False, scale=True)
        col_names = [f'Axis{i + 1}' for i in range(tab.shape[1])]
        tab.columns = col_names

        # Assign relevant values to mcoin
        mcoin['Tlw'] = ktcoa['lw']
        mcoin['Tcw'] = ktcoa['cw']
        mcoin['blo'] = ktcoa['blo']
        mcoin['Tc1'] = tab
        mcoin['RV'] = RV

        # Return results
        mciares = {'mcoa': mcoin, 'coa': nsca_results_list}
        return mciares


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

    # Calculate the number of zero-filled columns required
    pcolzero = nf2 - nf1 + 1

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


def normalize_per_block(scorcol, nbloc, indicablo, veclev, tol=1e-7):
    """
    Normalize `scorcol` by block, based on the block indicators `indicablo`
    and the unique block levels `veclev`.

    Parameters:
    - scorcol: np.array, scores or values to be normalized.
    - nbloc: int, number of blocks.
    - indicablo: np.array, block indicators for each element in `scorcol`.
    - veclev: np.array, unique block levels.
    - tol: float, tolerance below which normalization is not performed.

    Returns:
    - np.array, the normalized `scorcol`.
    """
    for i in range(nbloc):
        block_values = scorcol[indicablo == veclev[i]]
        block_norm = np.sqrt(np.sum(block_values ** 2))

        if block_norm > tol:
            block_values /= block_norm

        scorcol[indicablo == veclev[i]] = block_values

    return scorcol


def recalculate(tab, scorcol, nbloc, indicablo, veclev):
    """
    Adjust values in `tab` based on `scorcol` by block.

    Parameters:
    - tab: pd.DataFrame, table to be adjusted.
    - scorcol: np.array, scores or values used for the adjustment.
    - nbloc: int, number of blocks.
    - indicablo: np.array, block indicators for columns in `tab`.
    - veclev: np.array, unique block levels.

    Returns:
    - pd.DataFrame, the adjusted table.
    """
    for k in range(nbloc):
        mask = indicablo.values == veclev[k]
        subtable = tab.loc[:, mask]
        u_values = scorcol[mask]

        sum_values = (subtable * u_values).sum(axis=1)
        adjusted_subtable = subtable.sub(np.outer(sum_values, u_values))

        tab.loc[:, mask] = adjusted_subtable

    return tab


def sepan(data, nf=2):
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

    auxi = as_dudi(data[0], col_w=cw[j1:j2], row_w=lw, nf=nf, scannf=False, type="sepan")

    if auxi['factor_numbers'] < nf:
        auxi = complete_dudi(auxi, auxi['factor_numbers'] + 1, nf)

    Eig = list(auxi['eigenvalues'])
    Co = auxi['principal_coordinates']
    Li = auxi['factor_scores']
    C1 = auxi['component_scores']
    L1 = auxi['row_coordinates']

    rank = []
    rank.append(auxi['rank'])

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
        auxi = as_dudi(tab, cw[j1:j2], lw, nf=nf, scannf=False)

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
    # Generate row names
    w = x['row.names']
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

    print(col_names)

    w = x['tab.names']
    l0 = len(w)
    tab_names = list()

    # Repeat the entire array 'w' 4 times
    # todo Check this with higher number of dataset
    for i in range(len(w)):
        for k in range(1, 5):
            tab_names.append(f"{w[i]}.{k}")

    print(tab_names)

    # Check for 'kcoinertia' class
    if 'kcoinertia' not in x['class']:
        return {'row': row_names, 'col': col_names, 'tab': tab_names}

    # For 'kcoinertia' class, generate Trow names
    # NOTE: Assumes 'supblo' and 'supX' will be defined when needed
    trow_names = [f"{i}.{j}" for i, count in zip(x['tab.names'], x['supblo']) for j in x['supX'][:count]]

    return {'row': row_names, 'col': col_names, 'tab': tab_names, 'Trow': trow_names}


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


def mcoa(X, option=None, nf=3, tol=1e-07):
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
    veclev = list(set(X['TC']['T']))

    Xsepan = sepan(X, nf=4)
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
        tabw = [1 / sums[i] for i in sorted(sums.keys())]
    elif option == "uniform":
        tabw = [1] * nbloc
    elif option == "internal":
        tabw = X['tabw']
    else:
        raise ValueError("Unknown option")

    for i in range(nbloc):
        X[i] = [X[i] * np.sqrt(tabw[i])]

    for k in range(nbloc):
        X[k] = pd.DataFrame(X[k][0])

    Xsepan = sepan(X, nf=4)  # Recalculate sepan with the updated X

    # Convert the first element of X to a DataFrame and assign it to tab
    tab = pd.DataFrame(X[0])

    # Concatenate the remaining elements of X (from the 2nd to nbloc) to the columns of tab
    for i in range(1, nbloc):
        tab = pd.concat([tab, pd.DataFrame(X[i])], axis=1)

    # Assign the names of the columns of tab from auxinames['col']
    tab.columns = auxinames['col']

    # Multiply the rows of tab by the square root of lw
    tab = tab.mul(np.sqrt(lw), axis=0)

    # Multiply the columns of tab by the square root of cw
    tab = tab.mul(np.sqrt(cw), axis=1)

    # Initialize empty lists compogene and uknorme, and a None value valsing
    compogene = []
    uknorme = []
    valsing = None

    # Set the value of nfprovi to the minimum of 20, nlig, and ncol
    nfprovi = min(20, nlig, ncol)

    for i in range(nfprovi):
        # Perform singular value decomposition (SVD) on tab
        u, s, vt = np.linalg.svd(tab)

        # Extract the first column of u and normalize by the square root of lw (row_weights)
        normalized_u = u[:, 0] / np.sqrt(lw)
        compogene.append(normalized_u)  # Append to the list of compogene

        # Extract the first column of vt (v transposed in SVD), then normalize by the custom function normalizer_per_block
        normalized_v = normalize_per_block(vt[0, :], nbloc, indicablo, veclev)

        # Re-calculate tab using the custom function recalculate
        tab = recalculate(tab, normalized_v, nbloc, indicablo, veclev)

        # Normalize normalized_v by the square root of cw (column_weights) and append to the list of uknorme
        normalized_v /= np.sqrt(cw)
        uknorme.append(normalized_v)

        # Extract the first singular value from s and append to valsing
        singular_value = np.array([s[0]])
        valsing = np.concatenate([valsing, singular_value]) if valsing is not None else singular_value

    # Squaring the valsing to get pseudo eigenvalues
    pseudo_eigenvalues = valsing ** 2

    # Ensure nf is at least 2
    if nf <= 0:
        nf = 2

    # Initialize a dictionary to store different components
    acom = {}
    acom['pseudo_eigenvalues'] = pseudo_eigenvalues
    rank_fac = np.array(rank_fac)
    # Initialize a matrix to store eigenvalues for different blocks
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
    syn_var_df = pd.DataFrame(np.array(compogene).T[:, :nf])
    syn_var_df.columns = [f'SynVar{i}' for i in range(1, nf + 1)]
    syn_var_df.index = X['row.names']
    acom['SynVar'] = syn_var_df

    # Create a DataFrame for axes
    axis_df = pd.DataFrame(np.array(uknorme).T[:, :nf])
    axis_df.columns = [f'Axis{i}' for i in range(1, nf + 1)]
    axis_df.index = auxinames['col']
    acom['axis'] = axis_df

    # Initialize matrices for further processing
    w = np.zeros((nlig * nbloc, nf))
    covar = np.zeros((nbloc, nf))
    i2 = 0
    current_index = 0  # starting from the first row of w

    # Iterate over blocks
    for k in range(nbloc):
        i1 = i2
        i2 = i2 + nlig
        mask = indicablo == veclev[k]
        urk = acom['axis'].reset_index(drop=True).loc[mask].values
        # Extract corresponding matrix from X
        tab = np.array(X[k])
        # Multiply urk by appropriate column weights
        cw_array = np.array(cw)
        urk *= cw_array[mask].reshape(-1, 1)
        urk = tab @ urk
        # Assign the rows of urk to w, starting from current_index
        rows_in_urk = urk.shape[0]
        w[current_index:current_index + rows_in_urk, :] = urk

        # Move the current_index forward by the number of rows just added
        current_index += rows_in_urk

        # Matrix multiplication with acom['SynVar'] and element-wise multiplication with lw
        urk = (urk * acom['SynVar'].values) * lw.reshape(-1, 1)
        covar[k, :] = urk.sum(axis=0)

    # Convert w to DataFrame with appropriate row names and column names
    w_df = pd.DataFrame(w, index=auxinames['row'])
    w_df.columns = [f"Axis{str(i + 1)}" for i in range(nf)]
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
    acom['Tl1'] = w_df

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
    indicablo_reset = indicablo.reset_index(drop=True)

    # Iterate over blocks to update w and var.names based on axis and Xsepan
    for k in range(nbloc):
        i1 = i2 + 1
        i2 = i2 + 4

        bool_filter = indicablo_reset == veclev[k]

        urk = acom['axis'].reset_index(drop=True)[bool_filter].values
        tab = Xsepan['component_scores'].reset_index(drop=True)[bool_filter].values
        bool_filter_array = np.array(bool_filter)
        filtered_cw = np.array([cw[i] for i, flag in enumerate(bool_filter_array) if flag]).reshape(-1, 1)
        urk = urk * filtered_cw
        tab = tab.T.dot(urk)
        tab = np.nan_to_num(tab) # todo Check that this does not give error. The result for two datasets works, but could be a hard coding. The concept is that this function produced NaN instead of 0.00000, must be checked.

        for i in range(min(nf, 4)):
            if tab[i, i] < 0:
                tab[i, :] = -tab[i, :]

        w[i1 - 1:i2, :] = tab
        var_names.extend([f"{Xsepan['tab_names'][k]}.a{str(i + 1)}" for i in range(4)])

    # Create DataFrame for w and store it as Tax in acom
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
