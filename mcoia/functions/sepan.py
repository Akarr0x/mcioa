import pandas as pd
import numpy as np
from .data_preprocessing import decompose_data_to_principal_coords
from .data_reformat import complete_dudi


def multi_block_eigenanalysis(data, num_factors=20):
    """
    Compute successive eigenanalysis on partitioned data for multi-block analysis.

    Parameters:
    - data (dict): A dictionary containing data, weights, block definitions, and other information.
    - num_factors (int): The number of factors to compute.

    Returns:
    - dict: Results containing coordinates, scores, eigenvalues, and additional information.
    """

    # Validate input data class
    if data.get('class') != "ktab":
        raise ValueError("Input data must be of class 'ktab'.")

    row_weights = data['row_weight']
    column_weights = data['column_weight']
    blocks = data['blocks']
    num_tables = len(blocks)

    start_index = 0
    end_index = list(blocks.values())[0]

    # Decompose first block
    initial_decomposition = decompose_data_to_principal_coords(data[0], col_w=column_weights[start_index:end_index],
                                                               row_w=row_weights, nf=num_factors, class_type="sepan")

    # Complete the decomposition if factors are less than requested
    if initial_decomposition['factor_numbers'] < num_factors:
        initial_decomposition = complete_dudi(initial_decomposition, initial_decomposition['factor_numbers'] + 1,
                                              num_factors)

    eigenvalues = list(initial_decomposition['eigenvalues'])
    column_principal_coords = initial_decomposition['column_principal_coordinates']
    row_scores = initial_decomposition['row_scores']
    column_scores = initial_decomposition['column_scores']
    row_principal_coords = initial_decomposition['row_principal_coordinates']

    decomposition_ranks = [initial_decomposition['rank']]

    # Mapping for coordinates and scores
    coords_scores_mapping = {
        'column_principal_coordinates': 'Co',
        'row_scores': 'Li',
        'column_scores': 'C1',
        'row_principal_coordinates': 'L1'
    }

    # Update indices for coordinates and scores
    for dataframe in [row_scores, row_principal_coords, column_principal_coords, column_scores]:
        dataframe.index = [f'{index}.{start_index}' for index in dataframe.index]

    for i, block_key in enumerate(list(blocks.keys())[1:], start=1):
        start_index = end_index
        end_index += blocks[block_key]
        current_block_data = data[i]
        block_decomposition = decompose_data_to_principal_coords(current_block_data,
                                                                 column_weights[start_index:end_index], row_weights,
                                                                 nf=num_factors)

        # Extend eigenvalues and concatenate dataframes
        eigenvalues.extend(block_decomposition['eigenvalues'].tolist())

        for key, short_name in coords_scores_mapping.items():
            temp_dataframe = block_decomposition[key].copy()
            temp_dataframe.index = [f'X{idx + 1}.{block_key}' for idx in range(len(temp_dataframe))]
            if short_name == 'Co':
                column_principal_coords = pd.concat([column_principal_coords, temp_dataframe])
            elif short_name == 'Li':
                row_scores = pd.concat([row_scores, temp_dataframe])
            elif short_name == 'C1':
                column_scores = pd.concat([column_scores, temp_dataframe])
            elif short_name == 'L1':
                row_principal_coords = pd.concat([row_principal_coords, temp_dataframe])

        # Complete decomposition for current block if needed
        if block_decomposition['factor_numbers'] < num_factors:
            block_decomposition = complete_dudi(block_decomposition, block_decomposition['factor_numbers'] + 1,
                                                num_factors)

        decomposition_ranks.append(block_decomposition['rank'])

    # Convert lists to numpy arrays
    eigenvalues_array = np.array(eigenvalues)
    decomposition_ranks_array = np.array(decomposition_ranks)

    # Aggregate results
    analysis_results = {
        'row_principal_coordinates': row_principal_coords,
        'column_scores': column_scores,
        'column_principal_coordinates': column_principal_coords,
        'row_scores': row_scores,
        'eigenvalues': eigenvalues_array,
        'TL': data['TL'],
        'TC': data['TC'],
        'T4': data['T4'],
        'blocks': blocks,
        'rank': decomposition_ranks_array,
        'tab_names': list(data.keys())[:num_tables],
        'class': ["sepan", "list"]
    }

    return analysis_results
