import numpy as np
import pandas as pd


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
    This function adds additional factors for easier managing.
    It creates three new factors based on the blocks, row names, and column names.

    :param ktab_dict: A dictionary representing a k-table
    :return: The k-table dictionary with added factors
    """

    block_sizes = list(ktab_dict['blocks'].values())
    num_rows = len(ktab_dict['row_weight'])
    num_blocks = len(ktab_dict['blocks'])
    row_names = ktab_dict['row.names']
    col_names = ktab_dict['col.names']
    block_names = sorted(ktab_dict['tab.names'], key=lambda x: int(x.replace('Ana', '')))

    T_factor = np.repeat(block_names, num_rows)
    L_factor = np.tile(row_names, num_blocks)
    TL_df = pd.DataFrame({'T': T_factor, 'L': L_factor}) # L stands for rows
    ktab_dict['TL'] = TL_df

    T_factor = np.repeat(block_names, block_sizes)
    C_factor = np.concatenate(col_names)

    TC_df = pd.DataFrame({'T': T_factor, 'C': C_factor}) # C for columns
    ktab_dict['TC'] = TC_df

    T_factor = np.repeat(block_names, 4)
    four_factor = np.tile(np.arange(1, 5), num_blocks)
    T4_df = pd.DataFrame({'T': T_factor, '4': four_factor})
    ktab_dict['T4'] = T4_df

    return ktab_dict


def compile_tables(objects, rownames=None, colnames=None):
    """
    The function compiles a list of tables (as dictionaries) with 'row_weight', 'column_weight', and 'weighted_table'
    attributes into a single output dictionary 'compiled_tables' to be passed to ktab_util_addfactor function.
    """

    if not all(('row_weight' in item and 'column_weight' in item and 'weighted_table' in item) for item in objects):
        raise ValueError("list of objects with 'row_weight', 'column_weight', and 'weighted_table' attributes expected")

    num_blocks = len(objects)
    compiled_tables = {}
    row_weights = objects[0]['row_weight']
    column_weights = []
    block_lengths = [item['weighted_table'].shape[1] for item in objects]
    tablenames = [f"Ana{idx + 1}" for idx in range(len(objects))]

    for idx in range(num_blocks):
        if not np.array_equal(objects[idx]['row_weight'], row_weights):
            raise ValueError("Non equal row weights among arrays")

        compiled_tables[idx] = objects[idx]['weighted_table']
        column_weights.extend(objects[idx]['column_weight'])

    colnames_all_objects = [item['weighted_table'].columns.tolist() for item in objects]

    if rownames is None:
        rownames = objects[0]['weighted_table'].index.tolist()
    elif len(rownames) != len(objects[0]['weighted_table'].index):
        raise ValueError("Non convenient rownames length")

    if colnames is None:
        colnames = colnames_all_objects
    elif len(colnames) != len(colnames_all_objects):
        raise ValueError("Non convenient colnames length")


    if tablenames is None:
        tablenames = tablenames
    elif len(tablenames) != len(tablenames):
        raise ValueError("Non convenient tablenames length")

    compiled_tables['blocks'] = dict(zip(tablenames, block_lengths))
    compiled_tables['row_weight'] = row_weights
    compiled_tables['column_weight'] = column_weights
    compiled_tables['class'] = 'ktab'
    compiled_tables['row.names'] = rownames
    compiled_tables['col.names'] = colnames
    compiled_tables['tab.names'] = tablenames

    compiled_tables = add_factor_to_ktab(compiled_tables)

    return compiled_tables


def ensure_dimensional_costistency(dataset, nf1, nf2):
    """
    Aligns the dataset's dimensions with the requested number of factors by adding zero-filled columns

    Parameters:
    - dudi (dict): The original DUDI result containing various DataFrame structures.
    - nf1 (int): The start of the new range of axes.
    - nf2 (int): The end of the new range of axes.

    Returns:
    - dict: The updated dataset result with additional zero-filled columns.
    """

    def create_zero_df(rows, nf_start, nf_end):
        return pd.DataFrame(np.zeros((rows, nf_end - nf_start + 1)),
                            columns=[f'Axis{i}' for i in range(nf_start, nf_end + 1)])

    dataset['row_scores'] = pd.concat([dataset['row_scores'],
                                       create_zero_df(dataset['row_scores'].shape[0], nf1, nf2)],
                                      axis=1)

    dataset['row_principal_coordinates'] = pd.concat([dataset['row_principal_coordinates'],
                                                      create_zero_df(dataset['row_principal_coordinates'].shape[0], nf1, nf2)],
                                                     axis=1)

    dataset['column_principal_coordinates'] = pd.concat([dataset['column_principal_coordinates'],
                                                         create_zero_df(dataset['column_principal_coordinates'].shape[0], nf1, nf2)],
                                                        axis=1)

    dataset['column_score'] = pd.concat([dataset['column_scores'],
                                         create_zero_df(dataset['column_scores'].shape[0], nf1, nf2)],
                                        axis=1)

    return dataset
