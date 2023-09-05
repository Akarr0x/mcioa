import pandas as pd
import numpy as np
from .data_preprocessing import as_dudi
from .data_reformat import complete_dudi


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