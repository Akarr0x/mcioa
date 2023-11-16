import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from .sepan import multi_block_eigenanalysis
from .mcoa_processing import ktab_util_names, normalize_matrix_by_block, recalculate

def multiple_coinertia_analysis(data_table, weighting_option=None, number_factors=3, is_data_projected = False):
    """
    Conducts Multiple Co-Inertia Analysis on a given data table.

    :param data_table: A dictionary-like data structure representing the data for analysis.
    :param weighting_option: A list of weighting options for the analysis. Defaults to ["inertia", "lambda1", "uniform", "internal"].
    :param num_factors: Number of factors to use in the analysis.
    :param is_data_projected: Boolean indicating if the data is already projected.
    :return: A dictionary containing the results of the multiple co-inertia analysis.
    """
    if weighting_option is None:
        weighting_option = ["inertia", "lambda1", "uniform", "internal"]
    if data_table.get('class') != "ktab":
        raise ValueError("Expected object of class 'ktab'")
    weighting_option = weighting_option[0]
    if weighting_option == "internal":
        if data_table.get('weight_table') is None:
            print("Internal weights not found: uniform weights are used")
            weighting_option = "uniform"

    row_weight = data_table['row_weight']
    if isinstance(row_weight, pd.Series):
        row_weight = np.array(row_weight).reshape(-1, 1)

    nlig = len(row_weight)
    column_weight = data_table['column_weight']
    ncol = len(column_weight)
    num_bloc = len(data_table['blocks'])
    block_indices = data_table['TC']['T']
    unique_block_levels = sorted(list(set(data_table['TC']['T'])))

    multi_block_eigen_data = multi_block_eigenanalysis(data_table, 4)  # This is used to calculate the component scores factor scores for each data

    rank_factor = list(np.repeat(range(1, num_bloc + 1), multi_block_eigen_data["rank"]))

    auxiliary_names = ktab_util_names(data_table)
    sums = {}

    if weighting_option == "lambda1":
        weight_table = [1 / multi_block_eigen_data["eigenvalues"][rank_factor[i - 1]][0] for i in range(1, num_bloc + 1)]
    elif weighting_option == "inertia":
        for rank, value in zip(rank_factor, multi_block_eigen_data['eigenvalues']):
            sums[rank] = sums.get(rank, 0) + value

        weight_table = [1 / sums[i] for i in sorted(sums.keys())]
    elif weighting_option == "uniform":
        weight_table = [1] * num_bloc
    elif weighting_option == "internal":
        weight_table = data_table['weight_table']
    else:
        raise ValueError("Unknown option")

    for i in range(num_bloc):
        data_table[i] = [data_table[i] * np.sqrt(weight_table[i])]  # Weighting the datasets according to the calculated eigenvalues

    for k in range(num_bloc):
        data_table[k] = pd.DataFrame(data_table[k][0])

    multi_block_eigen_data = multi_block_eigenanalysis(data_table, 4)  # Recalculate sepan with the updated X

    # Convert the first element of X to a DataFrame and assign it to merged_datasets
    merged_datasets = pd.DataFrame(data_table[0])

    if is_data_projected:
        return data_table[0]

    # Concatenate the remaining elements of X (from the 2nd to num_bloc) to the columns of merged_datasets
    for i in range(1, num_bloc):
        merged_datasets = pd.concat([merged_datasets, pd.DataFrame(data_table[i])], axis=1)

    '''
    This creates the merged table of K weighted datasets
    '''

    # Assign the names of the columns of merged_datasets from auxiliary_names['col']
    merged_datasets.columns = auxiliary_names['col']

    merged_datasets = merged_datasets.mul(np.sqrt(row_weight), axis=0)
    merged_datasets = merged_datasets.mul(np.sqrt(column_weight), axis=1)

    vt_normalized_list = []
    singular_value_list = []

    nfprovi = min(20, nlig, ncol)
    syn_var_matrix = np.zeros((len(row_weight), nfprovi))

    # Perform SVD computations
    for i in range(nfprovi):

        truncated_svd = TruncatedSVD(n_components=nfprovi)
        u = truncated_svd.fit_transform(merged_datasets)
        s = truncated_svd.singular_values_
        vt = truncated_svd.components_
        u = u / s

        u_column = u[:, 0].flatten()

        row_weight = row_weight.values.flatten() if isinstance(row_weight, pd.Series) else row_weight.flatten()

        normalized_u = u_column / np.sqrt(row_weight)

        syn_var_matrix[:, i] = normalized_u

        normalized_v = normalize_matrix_by_block(vt[0, :], num_bloc, block_indices, unique_block_levels)

        merged_datasets = recalculate(merged_datasets, normalized_v, num_bloc, block_indices, unique_block_levels)

        normalized_v /= np.sqrt(column_weight)
        vt_normalized_list.append(normalized_v)

        singular_value = np.array([s[0]])

        singular_value_list = np.concatenate([singular_value_list, singular_value]) if singular_value_list is not None else singular_value
    pseudo_eigenvalues = singular_value_list ** 2

    if number_factors <= 0:
        number_factors = 2

    analysis_result = {'pseudo_eigenvalues': pseudo_eigenvalues}
    rank_factor = np.array(rank_factor)
    lambda_matrix = np.zeros((num_bloc, number_factors))
    for i in range(1, num_bloc + 1):
        mask = (rank_factor == i)

        w1 = multi_block_eigen_data['eigenvalues'][mask]

        r0 = multi_block_eigen_data['rank'][i - 1]
        if r0 > number_factors:
            r0 = number_factors

        lambda_matrix[i - 1, :r0] = w1[:r0]

    lambda_df = pd.DataFrame(lambda_matrix)
    lambda_df.index = multi_block_eigen_data['tab_names']
    lambda_df.columns = [f'lam{i}' for i in range(1, number_factors + 1)]
    analysis_result['lambda'] = lambda_df

    syn_var_df = pd.DataFrame(syn_var_matrix[:, :number_factors])
    syn_var_df.columns = [f'SynVar{i}' for i in range(1, number_factors + 1)]
    syn_var_df.index = data_table['row.names']
    analysis_result['SynVar'] = syn_var_df

    axis_df = pd.DataFrame(np.array(vt_normalized_list).T[:, :number_factors])
    axis_df.columns = [f'Axis{i}' for i in range(1, number_factors + 1)]
    axis_df.index = auxiliary_names['col']
    analysis_result['axis'] = axis_df

    w = np.zeros((nlig * num_bloc, number_factors))

    covar = np.zeros((num_bloc, number_factors))
    i2 = 0
    current_index = 0

    for k in range(num_bloc):
        i2 = i2 + nlig

        mask = block_indices == unique_block_levels[k]

        vk = analysis_result['axis'].reset_index(drop=True).loc[mask].values
        merged_datasets = np.array(data_table[k])

        cw_array = np.array(column_weight)
        vk *= cw_array[mask].reshape(-1, 1)

        projection = merged_datasets @ vk

        rows_in_projection = projection.shape[0]
        w[current_index:current_index + rows_in_projection, :] = projection
        current_index += rows_in_projection

        scaled_data = (projection * analysis_result['SynVar'].values) * row_weight.reshape(-1, 1)

        covar[k, :] = scaled_data.sum(axis=0)

    w_df = pd.DataFrame(w, index=auxiliary_names['row'])
    w_df.columns = [f"Axis{str(i + 1)}" for i in range(number_factors)]

    analysis_result['row_projection'] = w_df

    covar_df = pd.DataFrame(covar)
    covar_df.index = data_table['tab.names']
    covar_df.columns = [f"cov2{str(i + 1)}" for i in range(number_factors)]
    analysis_result['cov2'] = covar_df ** 2

    w = np.zeros((nlig * num_bloc, number_factors))
    i2 = 0
    for k in range(num_bloc):
        i1 = i2 + 1
        i2 = i2 + nlig
        merged_datasets = analysis_result['row_projection'].iloc[i1 - 1:i2, :]
        lw_sqrt = np.sqrt(row_weight)
        squared_values = (merged_datasets.values * lw_sqrt.reshape(-1, 1)) ** 2
        column_sums_sqrt = np.sqrt(squared_values.sum(axis=0))
        merged_datasets = merged_datasets.divide(column_sums_sqrt)
        w[i1 - 1:i2, :] = merged_datasets.values

    # Create DataFrame for adjusted w and store it as Tl1 in analysis_result
    w_df = pd.DataFrame(w, index=auxiliary_names['row'])
    w_df.columns = [f"Axis{str(i + 1)}" for i in range(number_factors)]
    analysis_result['row_projection_normed'] = w_df  # a normalized and re-scaled version of Tli

    w = np.zeros((ncol, number_factors))
    i2 = 0
    for k in range(num_bloc):
        i1 = i2 + 1
        i2 = i2 + data_table[k].shape[1]
        urk = np.array(analysis_result['SynVar'])
        merged_datasets = np.array(data_table[k])
        urk = urk * row_weight[:, np.newaxis]
        w[i1 - 1:i2, :] = merged_datasets.T.dot(urk)

    # Create DataFrame for w and store it as Tco in analysis_result
    w_df = pd.DataFrame(w, index=auxiliary_names['col'])
    w_df.columns = [f"SV{str(i + 1)}" for i in range(number_factors)]
    analysis_result['column_projection'] = w_df

    # Reset variables and initialize var.names
    var_names = []
    w = np.zeros((num_bloc * 4, number_factors))
    i2 = 0
    block_indices = block_indices.reset_index(drop=True)

    # Iterate over blocks to update w and var.names based on axis and multi_block_eigen_data
    for k in range(num_bloc):
        i1 = i2 + 1
        i2 = i2 + 4

        bool_filter = block_indices == unique_block_levels[k]

        urk = analysis_result['axis'].reset_index(drop=True)[bool_filter].values
        merged_datasets = multi_block_eigen_data['column_scores'].reset_index(drop=True)[bool_filter].values
        bool_filter_array = np.array(bool_filter)
        filtered_cw = np.array([column_weight[i] for i, flag in enumerate(bool_filter_array) if flag]).reshape(-1, 1)
        urk = urk * filtered_cw
        merged_datasets = merged_datasets.T.dot(urk)
        merged_datasets = np.nan_to_num(
            merged_datasets)

        for i in range(min(number_factors, 4)):
            if merged_datasets[i, i] < 0:
                merged_datasets[i, :] = -merged_datasets[i, :]

        w[i1 - 1:i2, :] = merged_datasets
        var_names.extend([f"{multi_block_eigen_data['tab_names'][k]}.a{str(i + 1)}" for i in range(4)])

    w_df = pd.DataFrame(w, index=auxiliary_names['tab'])
    w_df.columns = [f"Axis{str(i + 1)}" for i in range(number_factors)]
    analysis_result['Tax'] = w_df

    # Set additional properties of analysis_result
    analysis_result['nf'] = number_factors
    analysis_result['TL'] = data_table['TL']
    analysis_result['TC'] = data_table['TC']
    analysis_result['T4'] = data_table['T4']
    analysis_result['class'] = 'mcoa'

    return analysis_result
