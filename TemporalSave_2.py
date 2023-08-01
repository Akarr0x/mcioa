import pandas as pd
import numpy as np
from numpy.linalg import svd
import itertools
import matplotlib as plt
from scipy.linalg import eigh


def get_data(dataset):
    if isinstance(dataset, np.ndarray):
        if np.isrealobj(dataset):
            dataset = pd.DataFrame(dataset)

    if isinstance(dataset, pd.DataFrame):
        numeric_df = dataset.select_dtypes(include=[np.number])
        if dataset.shape[1] != numeric_df.shape[1]:
            print("Array data was found to be a data.frame, but contains non-numeric columns.")
            exit(1)
    return dataset


def Array2Ade4(dataset, pos=False, trans=False):
    if not (isinstance(dataset, pd.DataFrame)):
        dataset = get_data(dataset)  # This in case it is not a dataframe TODO think of other classes and add those

    for i in range(len(dataset)):
        if dataset[i].isnull().values.any():
            print(
                "Array data must not contain NA values. Use impute.knn in library(impute), KNNimpute from Troyanskaya et al., 2001 or LSimpute from Bo et al., 2004 to impute missing values\n")
            exit(1)

    if pos:
        for i in range(len(dataset)):
            if dataset[i].values.any() < 0:
                num = round(dataset[i].values.min()) - 1
                dataset[i] += abs(num)
    if trans:
        for i in range(len(dataset)):
            dataset[i] = dataset[i].T
        if not (isinstance(dataset, pd.DataFrame)):
            print("Problems transposing the dataset")
            exit(1)
    return dataset


from scipy.linalg import svd


def dudi_nsc(df, nf=2):
    df = pd.DataFrame(df)
    col = df.shape[1]

    if (df.values < 0).any():
        raise ValueError("negative entries in table")

    N = df.values.sum()
    if N == 0:
        raise ValueError("all frequencies are zero")

    row_w = df.sum(axis=1) / N
    col_w = df.sum(axis=0) / N

    # Transpose if more rows than columns
    transpose = False
    if df.shape[1] > df.shape[0]:
        transpose = True
        df = df.T
        col, row_w, col_w = df.shape[1], col_w, row_w  # Swap row and column weights

    df = df.T.apply(lambda x: col_w if x.sum() == 0 else x / x.sum()).T
    df = df.subtract(col_w, axis=1)
    df *= col

    X = as_dudi(df, np.ones(col) / col, row_w, nf)
    X['N'] = N

    return X


def as_dudi(df, col_w, row_w, nf=2, scannf=False, full=False, tol=1e-7, type = None):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Expected input is a pandas DataFrame.")

    lig, col = df.shape
    if len(col_w) != col:
        raise ValueError("Weights dimensions must match DataFrame dimensions.")
    if len(row_w) != lig:
        raise ValueError("Weights dimensions must match DataFrame dimensions.")
    if any(col_w < 0):
        raise ValueError("Weights must be non-negative.")
    if any(row_w < 0):
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

    eig_values, eig_vectors = eigh(eigen_matrix)
    eig_values = eig_values[::-1]

    rank = sum((eig_values / eig_values[0]) > tol)
    nf = min(nf, rank)

    if full:
        nf = rank

    res['eigenvalues'] = eig_values[:rank]
    res['rank'] = rank
    res['factor_numbers'] = nf
    col_w[col_w == 0] = 1
    row_w[row_w == 0] = 1
    dval = np.sqrt(res['eigenvalues'][:nf])

    if not transpose:
        col_w_sqrt_rec = 1 / np.sqrt(col_w)
        component_scores = eig_vectors[:, -nf:] * col_w_sqrt_rec.reshape(-1, 1)
        factor_scores = df_ori.multiply(res['column_weight'], axis=1)
        factor_scores = pd.DataFrame(factor_scores.values @ component_scores)  # Matrix multiplication and conversion to DataFrame

        res['component_scores'] = pd.DataFrame(component_scores, columns=[f'CS{i + 1}' for i in range(nf)])
        res['factor_scores'] = factor_scores
        res['factor_scores'].columns = [f'Axis{i + 1}' for i in range(nf)]
        res['principal_coordinates'] = res['component_scores'].multiply(dval[::-1])
        res['row_coordinates'] = res['factor_scores'].div(dval[::-1])
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
        res['class'] = 'dudi'
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
    # Define an internal function, rv, which calculates the RV coefficient between two datasets.

    # Get the number of datasets in dataset
    n = len(dataset)
    # Generate all combinations of 2 from the dataset names
    # For each combination, call the rv function with the 'weighted_table' of the corresponding datasets.
    # Store the resulting RV coefficients in RV.
    RV = [rv(dataset[name1]['weighted_table'].values, dataset[name2]['weighted_table'].values)
          for name1, name2 in itertools.combinations(dataset.keys(), 2)]

    # Create a nxn matrix filled with 1s
    m = np.ones((n, n))
    # Fill the lower triangle of m with the RV coefficients from RV
    m[np.tril_indices(n, -1)] = RV
    # Mirror the lower triangle to the upper triangle of m to make m symmetric
    m[np.triu_indices(n, 1)] = RV

    # Convert the numpy array m to a pandas DataFrame for easier manipulation and pretty printing
    m = pd.DataFrame(m)
    # Assign the names of the datasets as the row and column names of m
    m.columns = m.index = list(dataset.keys())
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
    res['ComponentScores'] = x['row_coordinates']
    res['RowCoordinates'] = x['component_scores']
    res['PrincipalCoordinates'] = x['factor_scores']
    res['FactorScores'] = x['principal_coordinates']
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


def mcia(dataset, nf=2, scan=False, nsc=True, svd=True):
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
    datasets_list = [np.array(df) for df in dataset]
    if len(set(total_columns)) != 1:
        print("Nonequal number of individuals across data frame")
        exit(1)
    for i, data in enumerate(dataset):
        if data.isnull().values.any():
            print("There are NA values")
            exit(1)
    if nsc:
        dataset = Array2Ade4(dataset, pos=True)

        # Perform Non-Symmetric Correspondence Analysis on each dataset
        nsca_results = {f'dataset_{i}': dudi_nsc(df, nf=nf) for i, df in enumerate(dataset)}
        nsca_results_t = nsca_results

        # Perform t_dudi on weighted_table of each nsca_result
        for name, result in nsca_results.items():
            nsca_results_t[name] = t_dudi(result)

        # Calculate the pairwise RV coefficients
        RV = pairwise_rv(nsca_results)
        nsca_results_list = list(nsca_results.values())
        ktcoa = compile_tables(nsca_results_list)

        # mcoin  = try(mcoa2(X = ktcoa, nf = cia.nf, scannf = FALSE), silent=TRUE)


def complete_dudi(dudi, nf1, nf2):
    pcolzero = nf2 - nf1 + 1
    w = pd.DataFrame(np.zeros((dudi['li'].shape[0], pcolzero)), columns=[f'Axis{i}' for i in range(nf1, nf2 + 1)])
    dudi['li'] = pd.concat([dudi['li'], w], axis=1)
    w = pd.DataFrame(np.zeros((dudi['li'].shape[0], pcolzero)), columns=[f'RS{i}' for i in range(nf1, nf2 + 1)])
    dudi['l1'] = pd.concat([dudi['l1'], w], axis=1)
    w = pd.DataFrame(np.zeros((dudi['co'].shape[0], pcolzero)), columns=[f'Comp{i}' for i in range(nf1, nf2 + 1)])
    dudi['co'] = pd.concat([dudi['co'], w], axis=1)
    w = pd.DataFrame(np.zeros((dudi['co'].shape[0], pcolzero)), columns=[f'CS{i}' for i in range(nf1, nf2 + 1)])
    dudi['c1'] = pd.concat([dudi['c1'], w], axis=1)
    return dudi


def normalizer_per_block(scorcol, nbloc, indicablo, veclev, tol=1e-7):
    for i in range(nbloc):
        w1 = scorcol[indicablo == veclev[i]]
        w2 = np.sqrt(np.sum(w1 * w1))
        if w2 > tol:
            w1 = w1 / w2
        scorcol[indicablo == veclev[i]] = w1
    return scorcol


def recalculate(tab, scorcol, nbloc, indicablo, veclev):
    for k in range(nbloc):
        soustabk = tab.loc[:, indicablo == veclev[k]]
        uk = scorcol[indicablo == veclev[k]]
        soustabk_hat = soustabk.apply(lambda x: np.sum(x * uk) * uk, axis=1)
        soustabk = soustabk - soustabk_hat
        tab.loc[:, indicablo == veclev[k]] = soustabk
    return tab


def sepan(X, nf=2):
    if X.get('class') != "ktab":
        raise ValueError("Expected object of class 'ktab'")

    lw = X['row_weight']
    cw = X['column_weight']
    blo = X['blocks']
    ntab = len(blo)
    j1 = 0
    j2 = blo[0]
    auxi = as_dudi(X[0], cw[j1:j2], lw, nf=nf, scannf=False, type = "sepan")

    if auxi['factor_numbers'] < nf:
        auxi = complete_dudi(auxi, auxi['factor_numbers'] + 1, nf)
    Eig = auxi['eigenvalues']
    Co = auxi['principal_coordinates'] #todo double check that this is correct, should be but the names given by the r function are deceiving
    Li = auxi['row_coordinates']
    C1 = auxi['component_scores']
    L1 = auxi['factor_scores']

    Li.index = [f'{index}.{j1}' for index in Li.index]
    L1.index = [f'{index}.{j1}' for index in L1.index]
    Co.index = [f'{index}.{j1}' for index in Co.index]
    C1.index = [f'{index}.{j1}' for index in C1.index]
    rank = auxi['rank']

    for i in range(1, ntab):
        j1 = j2
        j2 = j2 + blo[i]
        tab = X[i]
        auxi = as_dudi(tab, cw[j1:j2], lw, nf=nf, scannf=False)
        Eig = Eig + auxi['eigenvalues']
        auxi['row_coordinates'].index = [f'{index}.{i}' for index in auxi['row_coordinates'].index]
        auxi['component_scores'].index = [f'{index}.{i}' for index in auxi['component_scores'].index]
        auxi['principal_coordinates'].index = [f'{index}.{i}' for index in auxi['principal_coordinates'].index]
        auxi['factor_scores'].index = [f'{index}.{i}' for index in auxi['factor_scores'].index]

        if auxi['factor_numbers'] < nf:
            auxi = complete_dudi(auxi, auxi['factor_numbers'] + 1, nf)
        Co = pd.concat([Co, auxi['principal_coordinates']], axis=0)
        Li = pd.concat([Li, auxi['row_coordinates']], axis=0)
        C1 = pd.concat([C1, auxi['component_scores']], axis=0)
        L1 = pd.concat([L1, auxi['factor_scores']], axis=0)
        rank = rank + auxi['rank']

    res = {}
    res['row_coordinates'] = Li
    res['component_scores'] = L1
    res['principal_coordinates'] = Co
    res['factor_scores'] = C1
    res['eigenvalues'] = Eig
    res['TL'] = X['TL']
    res['TC'] = X['TC']
    res['T4'] = X['T4']
    res['blocks'] = blo
    res['rank'] = rank
    res['tab_names'] = list(X.keys())[:ntab]
    res['class'] = ["sepan", "list"]

    return res


def ktab_util_names(x):
    w = list(x['data'].keys())
    w1 = [f"{i}.{j}" for i, j in zip(w, x['TL'][0])]

    w = list(x['data'][w[0]].keys())
    if len(w) != len(set(w)):
        w = [f"{i}.{j}" for i, j in zip(w, x['TC'][0])]
    w2 = w

    w = x['tab.names']
    l0 = len(w)
    w3 = [f"{i}.{j}" for i in w for j in range(1, 5)]

    if 'kcoinertia' not in x['class']:
        return {'row': w1, 'col': w2, 'tab': w3}

    w4 = [f"{i}.{j}" for i, count in zip(x['tab.names'], x['supblo']) for j in x['supX'][:count]]

    return {'row': w1, 'col': w2, 'tab': w3, 'Trow': w4}

def tab_names(x, value):
    # Check if x has the required attributes for a 'ktab' object
    if not hasattr(x, 'blocks'):
        raise ValueError("to be used with 'ktab' object")

    ntab = len(x['blocks'])
    old = x['tab.names'][:ntab] if 'tab.names' in x else None

    if old is not None and len(value) != len(old):
        raise ValueError("invalid tab.names length")

    value = [str(v) for v in value]

    if len(set(value)) != len(value):
        raise ValueError("duplicate tab.names are not allowed")

    x['tab.names'] = value[:ntab]

    return x



def mcoa(X, option=None, scannf=True, nf=3, tol=1e-07):
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
    indicablo = X['TC'][0]
    veclev = list(set(X['TC'][0]))
    Xsepan = sepan(X, nf=4)
    rank_fac = list(np.repeat(range(1, nbloc + 1), Xsepan["rank"]))

    tabw = []
    auxinames = ktab_util_names(X)  # Assuming ktab_util_names is a function you've defined
    if option == "lambda1":
        tabw = [1 / Xsepan["eigenvalues"][rank_fac[i - 1]][0] for i in range(1, nbloc + 1)]
    elif option == "inertia":
        tabw = [1 / sum(Xsepan["eigenvalues"][rank_fac[i - 1]]) for i in range(1, nbloc + 1)]
    elif option == "uniform":
        tabw = [1] * nbloc
    elif option == "internal":
        tabw = X['tabw']
    else:
        raise ValueError("Unknown option")

    for i in range(nbloc):
        X[i] = [x * np.sqrt(tabw[i]) for x in X[i]]

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
    # Loop from 1 to nfprovi (as defined earlier)
    for i in range(nfprovi):
        # Perform singular value decomposition (SVD) on tab
        u, s, vt = np.linalg.svd(tab)

        # Extract the first column of u and normalize by the square root of lw (row_weights)
        normalized_u = u[:, 0] / np.sqrt(lw)
        compogene.append(normalized_u)  # Append to the list of compogene

        # Extract the first column of vt (v transposed in SVD), then normalize by the custom function normalizer_per_block
        normalized_v = normalizer_per_block(vt[0, :])

        # Re-calculate tab using the custom function recalculate
        tab = recalculate(tab, normalized_v)

        # Normalize normalized_v by the square root of cw (column_weights) and append to the list of uknorme
        normalized_v /= np.sqrt(cw)
        uknorme.append(normalized_v)

        # Extract the first singular value from s and append to valsing
        singular_value = s[0]
        valsing = valsing if valsing is None else np.concatenate([valsing, [singular_value]])

        # Squaring the valsing to get pseudo eigenvalues
        pseudo_eigenvalues = valsing ** 2

        # Ensure nf is at least 2
        if nf <= 0:
            nf = 2

        # Initialize a dictionary to store different components
        acom = {}
        acom['pseudo_eigenvalues'] = pseudo_eigenvalues

        # Initialize a matrix to store eigenvalues for different blocks
        lambda_matrix = np.zeros((nbloc, nf))
        for i in range(nbloc):
            w1 = Xsepan['eigenvalues'][rank_fac == i]
            r0 = Xsepan['rank'][i]
            if r0 > nf:
                r0 = nf
            lambda_matrix[i, :r0] = w1[:r0]

        # Convert the matrix to a DataFrame and assign row and column names
        lambda_df = pd.DataFrame(lambda_matrix)
        lambda_df.index = Xsepan['tab_names']
        lambda_df.columns = [f'lam{i}' for i in range(1, nf + 1)]
        acom['lambda'] = lambda_df

        # Create a DataFrame for synthesized variables
        syn_var_df = pd.DataFrame(np.array(compogene).T[:, :nf])
        syn_var_df.columns = [f'SynVar{i}' for i in range(1, nf + 1)]
        syn_var_df.index = X.index.names
        acom['SynVar'] = syn_var_df

        # Create a DataFrame for axes
        axis_df = pd.DataFrame(np.array(uknorme).T[:, :nf])
        axis_df.columns = [f'Axis{i}' for i in range(1, nf + 1)]
        axis_df.index = auxinames['col']
        acom['axis'] = axis_df

        # Initialize matrices for further processing
        w = np.zeros((nlig * nbloc, nf))
        covar = np.zeros((nbloc, nf))
        i1 = 0
        i2 = 0

        # Initialize variables
        i1 = 0
        i2 = 0
        w = np.zeros((nlig * nbloc, nfprovi))
        covar = np.zeros((nbloc, nfprovi))

        # Iterate over blocks
        for k in range(nbloc):
            i1 = i2 + 1
            i2 = i2 + nlig
            # Select appropriate rows from the axis DataFrame
            urk = acom['axis'].loc[indicablo == veclev[k]].values
            # Extract corresponding matrix from X
            tab = np.array(X[k])
            # Multiply urk by appropriate column weights
            urk = urk * cw[indicablo == veclev[k]]
            urk = tab.dot(urk)
            w[i1 - 1:i2, :] = urk
            urk = urk.dot(acom['SynVar']) * lw
            covar[k, :] = urk.sum(axis=1)

        # Convert w to DataFrame with appropriate row names and column names
        w_df = pd.DataFrame(w, index=auxinames['row'])
        w_df.columns = [f"Axis{str(i + 1)}" for i in range(nfprovi)]
        acom['Tli'] = w_df

        # Convert covar to DataFrame and square it, then store in acom
        covar_df = pd.DataFrame(covar)
        covar_df.index = tab_names(X)  # Assuming tab_names is a function that returns the appropriate row names
        covar_df.columns = [f"cov2{str(i + 1)}" for i in range(nfprovi)]
        acom['cov2'] = covar_df ** 2

        # Reset i1 and i2
        i1 = 0
        i2 = 0
        # Initialize w and indices
        w = np.zeros((nlig * nbloc, nfprovi))
        i1 = 0
        i2 = 0

        # Iterate over blocks to adjust w based on Tli and sqrt of lw
        for k in range(nbloc):
            i1 = i2 + 1
            i2 = i2 + nlig
            tab = acom['Tli'].iloc[i1 - 1:i2, :]
            adjusted_tab = (tab * np.sqrt(lw)).pow(2).sum(axis=0).apply(np.sqrt)
            tab = tab.divide(adjusted_tab, axis=1)
            w[i1 - 1:i2, :] = tab

        # Create DataFrame for adjusted w and store it as Tl1 in acom
        w_df = pd.DataFrame(w, index=auxinames['row'])
        w_df.columns = [f"Axis{str(i + 1)}" for i in range(nfprovi)]
        acom['Tl1'] = w_df

        # Reset variables
        w = np.zeros((ncol, nfprovi))
        i1 = 0
        i2 = 0

        # Iterate over blocks to update w based on SynVar and lw
        for k in range(nbloc):
            i1 = i2 + 1
            i2 = i2 + X[k].shape[1]
            urk = np.array(acom['SynVar'])
            tab = np.array(X[k])
            urk = urk * lw
            w[i1 - 1:i2, :] = tab.T.dot(urk)

        # Create DataFrame for w and store it as Tco in acom
        w_df = pd.DataFrame(w, index=auxinames['col'])
        w_df.columns = [f"SV{str(i + 1)}" for i in range(nfprovi)]
        acom['Tco'] = w_df

        # Reset variables and initialize var.names
        var_names = []
        w = np.zeros((nbloc * 4, nfprovi))
        i1 = 0
        i2 = 0

        # Iterate over blocks to update w and var.names based on axis and Xsepan
        for k in range(nbloc):
            i1 = i2 + 1
            i2 = i2 + 4
            urk = acom['axis'].loc[indicablo == veclev[k]].values
            tab = Xsepan['C1'].loc[indicablo == veclev[k]].values
            urk = urk * cw[indicablo == veclev[k]]
            tab = tab.T.dot(urk)
            for i in range(min(nfprovi, 4)):
                if tab[i, i] < 0:
                    tab[i, :] = -tab[i, :]
            w[i1 - 1:i2, :] = tab
            var_names.extend([f"{Xsepan['tab.names'][k]}.a{str(i + 1)}" for i in range(4)])

        # Create DataFrame for w and store it as Tax in acom
        w_df = pd.DataFrame(w, index=auxinames['tab'])
        w_df.columns = [f"Axis{str(i + 1)}" for i in range(nfprovi)]
        acom['Tax'] = w_df

        # Set additional properties of acom
        acom['nf'] = nf
        acom['TL'] = X['TL']
        acom['TC'] = X['TC']
        acom['T4'] = X['T4']
        acom['class'] = 'mcoa'

        # Assuming match.call() equivalent is needed in Python
        acom['call'] = "Equivalent of match.call() in Python"

        return acom

