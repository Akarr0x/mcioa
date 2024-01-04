import pandas as pd
import numpy as np


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
    dudi['row_scores'] = pd.concat([dudi['row_scores'],
                                       create_zero_df(dudi['row_scores'].shape[0], nf1, nf2)],
                                      axis=1)

    # Extend 'row_coordinates' with zero-filled columns
    dudi['row_principal_coordinates'] = pd.concat([dudi['row_principal_coordinates'],
                                         create_zero_df(dudi['row_principal_coordinates'].shape[0], nf1, nf2)],
                                        axis=1)

    # Extend 'principal_coordinates' with zero-filled columns
    dudi['column_principal_coordinates'] = pd.concat([dudi['column_principal_coordinates'],
                                               create_zero_df(dudi['column_principal_coordinates'].shape[0], nf1, nf2)],
                                              axis=1)

    # Extend 'component_scores' with zero-filled columns
    dudi['column_score'] = pd.concat([dudi['column_scores'],
                                          create_zero_df(dudi['column_scores'].shape[0], nf1, nf2)],
                                         axis=1)

    return dudi
