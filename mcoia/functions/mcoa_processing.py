import pandas as pd
import numpy as np


def normalize_matrix_by_block(Vt, num_blocks, block_indicator, unique_block_levels, tol=1e-7):
    """
    Normalize `scorcol` by block, based on the block indicators `indicablo`
    and the unique block levels `veclev`.
    This function is used to be sure that u_k is unitary

    Parameters:
    - scores: np.array, scores or values to be normalized.
    - num_blocks: int, number of blocks.
    - block_indicator: np.array, block indicators for each element in `scorcol`.
    - unique_block_levels: np.array, unique block levels.
    - tol: float, tolerance below which normalization is not performed.

    Returns:
    - np.array, the normalized `scores`.
    """
    for i in range(num_blocks):
        block_values = Vt[block_indicator == unique_block_levels[i]]
        block_norm = np.sqrt(np.sum(block_values ** 2))

        if block_norm > tol:
            block_values /= block_norm

        Vt[block_indicator == unique_block_levels[i]] = block_values

    return Vt


def recalculate(dataframe, vt, num_blocks, column_indicator, unique_levels):
    """
    Adjust values in `dataframe` based on `vt` for each block.
    This function implements the deflation method used in multiple correspondence analysis.


    Parameters:
    - dataframe: pd.DataFrame, table to be adjusted.
    - vt: np.array, scores or values used for the adjustment.
    - num_blocks: int, number of blocks.
    - column_indicator: np.array, block indicators for columns in `tab`.
    - unique_levels: np.array, unique block levels.

    Returns:
    - pd.DataFrame, the adjusted dataframe.
    """
    dataframe_values = dataframe.values
    indicablo_np = column_indicator.values
    for k in range(num_blocks):
        mask = indicablo_np == unique_levels[k]
        subtable = dataframe_values[:, mask]
        u_values = vt[mask]

        sum_values = (subtable * u_values).sum(axis=1)
        adjusted_subtable = subtable - np.outer(sum_values, u_values)

        dataframe_values[:, mask] = adjusted_subtable

    return pd.DataFrame(dataframe_values, columns=dataframe.columns)


def ktab_util_names(ktab_object):
    """
    Generates utility names for ktab objects, including row, column, and tab names

    Parameters:
    - ktab_object (dict): The ktab object with keys 'data', 'TL', 'TC', 'tab.names', and 'class'.

    Returns:
    - dict: A dictionary containing utility names.
    """
    time_level_suffixes = [f"df{i + 1}" for i, time_level in enumerate(ktab_object['TL']['T'].unique())]
    row_utility_names = []

    for index, time_level in enumerate(ktab_object['TL']['T'].unique()):
        time_level_subset = ktab_object['TL'][ktab_object['TL']['T'] == time_level]
        row_utility_names.extend([f"{level}.{time_level_suffixes[index]}" for level in time_level_subset['L']])

    column_secondary_keys = ktab_object['col.names']
    unique_suffixes = [f"{str(i)}" for i in set(ktab_object['TC']['T'])]

    column_utility_names = []
    for key_list, suffix in zip(column_secondary_keys, unique_suffixes):
        column_utility_names.extend([f"{key}.{suffix}" for key in key_list])

    tab_names = ktab_object['tab.names']
    tab_utility_names = [f"{name}.{iteration}" for name in tab_names for iteration in range(1, 5)]

    if 'kcoinertia' in ktab_object['class']:
        trow_names = [f"{tab_name}.{sub_index}" for tab_name, block_count in zip(tab_names, ktab_object['supblo']) for sub_index in ktab_object['supX'][:block_count]]
        return {'row': row_utility_names, 'col': column_utility_names, 'tab': tab_utility_names, 'Trow': trow_names}

    return {'row': row_utility_names, 'col': column_utility_names, 'tab': tab_utility_names}



def update_ktab_tab_names(ktab_object, new_tab_names):
    """
    Assign or modify the 'tab.names' attribute of a 'ktab' (Knowledge Tabulation) object.

    Parameters:
    - ktab_object (dict): The ktab object expected to contain a 'blocks' key.
    - new_tab_names (list): The list of new tab names to be assigned.

    Returns:
    - dict: The ktab object updated with the new 'tab.names' attribute.

    Raises:
    - ValueError: If `ktab_object` is not a valid ktab object, if the length of provided tab names is incorrect,
                  or if there are duplicate tab names.
    """
    if 'blocks' not in ktab_object:
        raise ValueError("Provided object should be a ktab object.")

    number_of_tabs = len(ktab_object['blocks'])
    existing_tab_names = ktab_object.get('tab.names', [None] * number_of_tabs)

    if len(new_tab_names) != number_of_tabs:
        raise ValueError("Number of new tab names must match the number of tabs in the ktab object.")

    new_tab_names_str = [str(name) for name in new_tab_names]
    if len(set(new_tab_names_str)) != len(new_tab_names_str):
        raise ValueError("Duplicate names are not permitted in tab names.")

    ktab_object['tab.names'] = new_tab_names_str

    return ktab_object
