import pandas as pd
import numpy as np


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
