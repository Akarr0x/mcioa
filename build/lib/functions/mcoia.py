import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from mcioa.functions.sepan import multi_block_eigenanalysis
from mcioa.functions.mcoa_processing import ktab_util_names, normalize_matrix_by_block, recalculate

def row_normed_column_projection(number_rows, number_Datasets, n_dim, row_weight, datasets, auxiliary_names, analysis_result, number_Columns):
    matrix = np.zeros((number_rows * number_Datasets, n_dim))
    i2 = 0
    for k in range(number_Datasets):
        i1 = i2 + 1
        i2 = i2 + number_rows
        row_projection = analysis_result['row_projection'].iloc[i1 - 1:i2, :]
        row_weight_squared = np.sqrt(row_weight)
        squared_values = (row_projection.values * row_weight_squared.reshape(-1, 1)) ** 2
        column_sums_sqrt = np.sqrt(squared_values.sum(axis=0))
        row_projection = row_projection.divide(column_sums_sqrt)
        matrix[i1 - 1:i2, :] = row_projection.values

    analysis_result['row_projection_normed'] = pd.DataFrame(
        matrix,
        index=auxiliary_names['row'],
        columns=[f"Axis{i + 1}" for i in range(n_dim)]
    )

    matrix = np.zeros((number_Columns, n_dim))
    i2 = 0
    for k in range(number_Datasets):
        i1 = i2 + 1
        i2 = i2 + datasets[k].shape[1]
        u = np.array(analysis_result['SynVar'])
        dataset = np.array(datasets[k])
        u = u * row_weight[:, np.newaxis]
        matrix[i1 - 1:i2, :] = dataset.T.dot(u)

    analysis_result['column_projection'] = pd.DataFrame(
        matrix,
        index=auxiliary_names['col'],
        columns=[f"SV{i + 1}" for i in range(n_dim)]
    )

    return analysis_result

def calculate_row_projection(number_rows, number_Datasets, n_dim, block_Indicator, dataset_index, analysis_result,
                       datasets, column_weight, row_weight, auxiliary_names):
    matrix = np.zeros((number_rows * number_Datasets, n_dim))
    covar = np.zeros((number_Datasets, n_dim))
    current_index = 0

    for k in range(number_Datasets):
        mask = block_Indicator == dataset_index[k]
        vk = analysis_result['axis'].reset_index(drop=True).loc[mask].values
        x_Tilde = np.array(datasets[k])

        cw_array = np.array(column_weight)
        vk *= cw_array[mask].reshape(-1, 1)

        projection = x_Tilde @ vk
        rows_in_projection = projection.shape[0]
        matrix[current_index:current_index + rows_in_projection, :] = projection
        current_index += rows_in_projection

        scaled_data = (projection * analysis_result['SynVar'].values) * row_weight.reshape(-1, 1)
        covar[k, :] = scaled_data.sum(axis=0)

    w_df = pd.DataFrame(matrix, index=auxiliary_names['row'])
    w_df.columns = [f"Axis{i + 1}" for i in range(n_dim)]
    analysis_result['row_projection'] = w_df

    covar_df = pd.DataFrame(covar ** 2, index=datasets['tab.names'], columns=[f"cov2{i + 1}" for i in range(n_dim)])
    analysis_result['cov2'] = covar_df

    return analysis_result


def create_analysis_dataframes(analysis_result, lambda_matrix, average_view_u, v_k, multi_block_eigen_data, X, auxiliary_names, n_dim):

    analysis_result['lambda'] = pd.DataFrame(lambda_matrix, index=multi_block_eigen_data['tab_names'],
                                             columns=[f'lam{i}' for i in range(1, n_dim + 1)])

    analysis_result['SynVar'] = pd.DataFrame(np.array(average_view_u).T[:, :n_dim], index=X['row.names'],
                                             columns=[f'SynVar{i}' for i in range(1, n_dim + 1)])

    analysis_result['axis'] = pd.DataFrame(np.array(v_k).T[:, :n_dim], index=auxiliary_names['col'],
                                           columns=[f'Axis{i}' for i in range(1, n_dim + 1)])


def multiple_coinertia_analysis(datasets, weight_option=None, n_dim=3, data_projected = False):
    if weight_option is None:
        weight_option = ["inertia", "lambda1", "uniform", "internal"]
    if datasets.get('class') != "ktab":
        raise ValueError("Expected object of class 'ktab'")
    weight_option = weight_option[0]
    if weight_option == "internal":
        if datasets.get('weight_table') is None:
            print("Internal weights not found: uniform weights are used")
            weight_option = "uniform"

    row_weight = datasets['row_weight']
    number_rows = len(row_weight)
    column_weight = datasets['column_weight']
    number_Columns = len(column_weight)
    number_Datasets = len(datasets['blocks'])
    block_Indicator = datasets['TC']['T']
    dataset_indix = sorted(list(set(datasets['TC']['T'])))

    multi_block_eigen_data = multi_block_eigenanalysis(datasets, nf=4)

    rank_per_block = list(np.repeat(range(1, number_Datasets + 1), multi_block_eigen_data["rank"]))

    auxiliary_names = ktab_util_names(datasets)
    sum_of_ranks = {}

    if weight_option == "lambda1":
        weight_table = [1 / multi_block_eigen_data["eigenvalues"][rank_per_block[i - 1]][0] for i in range(1, number_Datasets + 1)]
    elif weight_option == "inertia":
        for rank, value in zip(rank_per_block, multi_block_eigen_data['eigenvalues']):
            sum_of_ranks[rank] = sum_of_ranks.get(rank, 0) + value

        weight_table = [1 / sum_of_ranks[i] for i in sorted(sum_of_ranks.keys())]
    elif weight_option == "uniform":
        weight_table = [1] * number_Datasets
    elif weight_option == "internal":
        weight_table = datasets['weight_table']
    else:
        raise ValueError("Unknown option")

    for i in range(number_Datasets):
        datasets[i] = [datasets[i] * np.sqrt(weight_table[i])]

    for k in range(number_Datasets):
        datasets[k] = pd.DataFrame(datasets[k][0])

    multi_block_eigen_data = multi_block_eigenanalysis(datasets, nf=4)

    x_Tilde = pd.DataFrame(datasets[0])

    if data_projected:
        return datasets[0]

    for i in range(1, number_Datasets):
        x_Tilde = pd.concat([x_Tilde, pd.DataFrame(datasets[i])], axis=1)

    x_Tilde.columns = auxiliary_names['col']

    x_Tilde = x_Tilde.mul(np.sqrt(row_weight), axis=0)
    x_Tilde = x_Tilde.mul(np.sqrt(column_weight), axis=1)

    average_View_U = []
    v_k_Normalized = []
    singular_value_list = None

    min_dimensions = min(n_dim, number_rows, number_Columns)
    for i in range(min_dimensions):

        truncated_svd = TruncatedSVD(n_components=min_dimensions)
        u = truncated_svd.fit_transform(x_Tilde)
        s = truncated_svd.singular_values_
        vt = truncated_svd.components_
        u = u / s

        normalized_u = u[:, 0] / np.sqrt(row_weight)
        average_View_U.append(normalized_u)

        normalized_v = normalize_matrix_by_block(vt[0, :], number_Datasets, block_Indicator, dataset_indix)

        x_Tilde = recalculate(x_Tilde, normalized_v, number_Datasets, block_Indicator, dataset_indix)

        normalized_v /= np.sqrt(column_weight)
        v_k_Normalized.append(normalized_v)

        singular_value = np.array([s[0]])

        singular_value_list = np.concatenate([singular_value_list, singular_value]) if singular_value_list is not None else singular_value

    n_dim = max(n_dim, 2)
    analysis_result = {'pseudo_eigenvalues': singular_value_list ** 2}
    rank_per_block_array = np.array(rank_per_block)
    lambda_matrix = np.zeros((number_Datasets, n_dim))

    for dataset_index in range(number_Datasets):
        is_current_dataset = (rank_per_block_array == dataset_index + 1)
        current_eigenvalues = multi_block_eigen_data['eigenvalues'][is_current_dataset]
        current_rank = min(multi_block_eigen_data['rank'][dataset_index], n_dim)
        lambda_matrix[dataset_index, :current_rank] = current_eigenvalues[:current_rank]


    create_analysis_dataframes(analysis_result, lambda_matrix, average_View_U, v_k_Normalized, multi_block_eigen_data, datasets,
                        auxiliary_names, n_dim)

    calculate_row_projection(number_rows, number_Datasets, n_dim, block_Indicator, dataset_indix, analysis_result, datasets,
                        column_weight, row_weight, auxiliary_names)

    row_normed_column_projection(number_rows, number_Datasets, n_dim, row_weight, datasets, auxiliary_names,
                        analysis_result, number_Columns
    )

    analysis_result['nf'] = n_dim
    analysis_result['TL'] = datasets['TL']
    analysis_result['TC'] = datasets['TC']
    analysis_result['T4'] = datasets['T4']
    analysis_result['class'] = 'mcoa'

    return analysis_result