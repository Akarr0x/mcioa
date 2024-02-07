import pandas as pd
import numpy as np
from .data_preprocessing import decompose_data_to_principal_coords
from .data_reformat import complete_dudi


def perform_decomposition(tab, column_weight, row_weight, nf, block_key=None):
    auxi = decompose_data_to_principal_coords(tab, col_w=column_weight, row_w=row_weight, nf=nf, class_type="sepan")
    if auxi['factor_numbers'] < nf:
        auxi = complete_dudi(auxi, auxi['factor_numbers'] + 1, nf)

    if block_key:
        for key in ['column_principal_coordinates', 'row_scores', 'column_scores', 'row_principal_coordinates']:
            df = auxi[key]
            df.index = [f'X{idx + 1}.{block_key}' for idx in range(len(df))]

    return auxi


def multi_block_eigenanalysis(data, nf=20):
    """
     Calculates eigenvalues for each dataset independently, used to give different weights to the dataset
     """

    Eig, rank = [], []
    Co, Li, C1, L1 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    j2 = 0
    for i, (block_key, size) in enumerate(data['blocks'].items()):
        j1, j2 = j2, j2 + size
        column_weight_segment = data['column_weight'][j1:j2]

        auxi = perform_decomposition(data[i], column_weight_segment, data['row_weight'], nf, block_key)

        Eig.extend(auxi['eigenvalues'])
        rank.append(auxi['rank'])

        for key, df in [('column_principal_coordinates', Co), ('row_scores', Li),
                        ('column_scores', C1), ('row_principal_coordinates', L1)]:
            auxi_df = auxi[key]
            df = pd.concat([df, auxi_df])

    res = {
        'eigenvalues': np.array(Eig),
        'rank': np.array(rank),
        'column_principal_coordinates': Co,
        'row_scores': Li,
        'column_scores': C1,
        'row_principal_coordinates': L1,
        'TL': data['TL'],
        'TC': data['TC'],
        'T4': data['T4'],
        'blocks': data['blocks'],
        'tab_names': list(data.keys())[:len(data['blocks'])],
        'class': ["sepan", "list"]
    }
    return res
