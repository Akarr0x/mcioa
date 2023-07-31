import pandas as pd
import numpy as np
from numpy.linalg import svd
import itertools


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
        dataset = get_data(dataset) # This in case it is not a dataframe TODO think of other classes and add those

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

    df = df.T.apply(lambda x: col_w if x.sum() == 0 else x / x.sum()).T
    df = df.mul(col_w, axis=1)
    df *= col

    X = as_dudi(df, np.repeat(1, col) / col, row_w, nf)
    X['N'] = N

    return X



def as_dudi(df, col_w, row_w, nf=2):
    """
    The function weights the input data by square roots of the row and column weights to maintain total variance
    and then computes SVD. The reduced-dimension representation of the input data and corresponding variables are
    returned.
    """
    # TODO this does not take into account the possibility of having ncol > nrows and the eventual transpose because of it. Should be updated (!transpose in the r function)
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Expected input is a pandas DataFrame.")

    # Check if weights vectors match the size of rows and columns of the DataFrame
    if len(col_w) != df.shape[1] or len(row_w) != df.shape[0]:
        raise ValueError("Weights dimensions must match DataFrame dimensions.")

    # Check for negative values in weights
    if any(col_w < 0) or any(row_w < 0):
        raise ValueError("Weights must be non-negative.")

    # Create a copy of the DataFrame to prevent changes to original data
    df = df.copy()

    # Convert weights to numpy arrays and take their square roots
    # This process is to ensure each row/column contributes equally to the total variance
    row_w_sqrt = np.sqrt(row_w) if isinstance(row_w, np.ndarray) else np.sqrt(row_w.to_numpy())
    col_w_sqrt = np.sqrt(col_w) if isinstance(col_w, np.ndarray) else np.sqrt(col_w.to_numpy())

    # Apply weights to the DataFrame
    df *= row_w_sqrt[:, None]  # Multiply rows by sqrt of row weights
    df *= col_w_sqrt[None, :]  # Multiply columns by sqrt of column weights

    # Perform SVD on the weighted DataFrame
    # U contains left singular vectors, s contains singular values, Vt contains right singular vectors
    U, s, Vt = np.linalg.svd(df)

    # Compute eigenvalues from singular values
    # In SVD, eigenvalues are square of singular values divided by degrees of freedom (n-1 for n data points)
    eig = s ** 2 / (df.shape[0] - 1)

    # Rank of eigenvalues indicates the number of non-negligible dimensions in the data
    # It's computed as number of eigenvalues significantly greater than zero (here, greater than 1e-7 of the first eigenvalue)
    rank = (eig / eig[0] > 1e-7).sum()

    # The number of factors (dimensions) to keep is minimum of user-input 'nf' and the rank
    nf = min(nf, rank)

    # The dval is the square roots of the nf largest eigenvalues
    # They're used for scaling the component scores and factor scores
    dval = np.sqrt(eig[:nf])

    # Compute the reciprocal of the square root of weights
    # These are used to scale the component scores and factor scores to get principal coordinates and row coordinates
    col_w_sqrt_rec = np.reciprocal(col_w_sqrt)
    row_w_sqrt_rec = np.reciprocal(row_w_sqrt)

    # Compute component scores as the product of right singular vectors (components of Vt)
    # and the reciprocal of the square root of column weights
    component_scores = pd.DataFrame((Vt.T[:, :nf] * col_w_sqrt_rec[:, None]),
                                    columns=[f'ComponentScore{i + 1}' for i in range(nf)])

    # Compute factor scores as the product of the weighted DataFrame and the component scores
    # This operation projects the original data onto the new component axes
    factor_scores = pd.DataFrame((df.values @ component_scores.values),
                                 columns=[f'FactorScore{i + 1}' for i in range(nf)])

    # Compute principal coordinates as the product of the component scores and dval
    # The principal coordinates represent the original variables in the reduced-dimension space
    principal_coordinates = component_scores.multiply(dval)

    # Compute row coordinates as the division of the factor scores by dval
    # The row coordinates represent the original observations in the reduced-dimension space
    row_coordinates = factor_scores.div(dval)

    # Return the results as a dictionary
    return {
        'eigenvalues': eig[:rank],  # The eigenvalues
        'rank': rank,  # The rank
        'factor_numbers': nf,  # Number of factors
        'weighted_table': df,  # The weighted table
        'column_weight': col_w,  # Column weights
        'row_weight': row_w,  # Row weights
        'ComponentScores': component_scores,  # How much each column contributes to each factor
        'FactorScores': factor_scores,  # How much each row contributes to each factor
        'PrincipalCoordinates': principal_coordinates,  # Principal coordinates of the columns in the new space defined by the factors
        'RowCoordinates': row_coordinates,  # Principal coordinates of the columns in the new space defined by the factors
    }



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
    res['ComponentScores'] = x['RowCoordinates']
    res['RowCoordinates'] = x['ComponentScores']
    res['PrincipalCoordinates'] = x['FactorScores']
    res['FactorScores'] = x['PrincipalCoordinates']
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




