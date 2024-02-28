import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from mcioa.functions.normalization_per_block import ktab_util_names, normalize_matrix_by_block, recalculate
from mcioa.functions.multi_eigenanalysis import multi_block_eigenanalysis


def row_normed_column_projection(number_rows, number_Datasets, n_dim, row_weight, datasets, auxiliary_names,
                                 analysis_result, number_Columns):
    matrix = np.zeros((number_rows * number_Datasets, n_dim))
    """
    Performs normalization and projection of row data and direct projection of column data across multiple datasets.

    This function first normalizes the rows of the analysis results based on the provided row weights and then projects 
    these normalized rows into a specified number of dimensions. 
    
    Parameters:
    - number_rows (int): The number of rows in each dataset.
    - number_Datasets (int): The total number of datasets being analyzed.
    - n_dim (int): The number of dimensions to project the data into.
    - row_weight (np.ndarray): An array of weights for each row in the datasets.
    - datasets (list of np.ndarray or pd.DataFrame): The list of datasets to be analyzed.
    - auxiliary_names (dict): A dictionary containing auxiliary information such as 'row' and 'col' names for indexing the result matrices.
    - analysis_result (dict): A dictionary where the analysis results, including 'row_projection' and the synthetic variables ('SynVar'), are stored.
    - number_Columns (int): The total number of columns across all datasets.

    Returns:
    - dict: The updated analysis_result dictionary containing two new keys: 'row_projection_normed' and 'column_projection',
     which store the normalized row projections and the direct column projections as pandas DataFrames.
    """

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
    """
    Calculates the projection of rows from multiple datasets onto and computes the squared covariances of these projections.

    Parameters:
    - number_rows (int): The number of rows in each dataset.
    - number_Datasets (int): The total number of datasets to be processed.
    - n_dim (int): The number of dimensions onto which the rows are projected.
    - block_Indicator (np.ndarray): An array indicating the block or dataset each axis belongs to.
    - dataset_index (list[int]): A list of indices specifying the dataset each row belongs to.
    - analysis_result (dict): A dictionary to store the analysis results, including 'axis', 'SynVar', 'row_projection', and 'cov2'.
    - datasets (list of np.ndarray or pd.DataFrame): The datasets to be analyzed.
    - column_weight (np.ndarray): An array of weights for the columns in the datasets.
    - row_weight (np.ndarray): An array of weights for the rows in the datasets.
    - auxiliary_names (dict): A dictionary containing auxiliary information, such as the names for rows ('row').

    Returns:
    - dict: The updated analysis_result dictionary containing 'row_projection', a DataFrame of the row projections, and 'cov2', a DataFrame of the squared covariances.

    The function updates the 'analysis_result' dictionary with two new keys: 'row_projection', storing the projections
    of dataset rows onto the specified dimensions, and 'cov2', storing the squared covariances of these projections.
    """
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


def create_analysis_dataframes(analysis_result, lambda_matrix, average_view_u, v_k, multi_block_eigen_data, X,
                               auxiliary_names, n_dim):
    """
    Creates and stores essential dataframes from analysis results into a given dictionary.

    This function processes and provides the lambda matrix, synthetic variable (SynVar) calculations, and axis projections into pandas DataFrames. Each DataFrame is indexed appropriately and contains a specified number of dimensions (columns). These dataframes are crucial for interpreting the results of the multivariate analysis, providing insights into the eigenvalues, synthetic variables, and axis projections derived from the analysis.

    Parameters:
    - analysis_result (dict): A dictionary where the resulting DataFrames will be stored.
    - lambda_matrix (np.ndarray): A matrix containing eigenvalues or similar metrics that quantify variance explained by each dimension.
    - average_view_u (np.ndarray): An array representing the average or synthetic view of the data, used to calculate synthetic center.
    - v_k (np.ndarray): Right singular vectors.
    - multi_block_eigen_data (dict): A dictionary containing metadata from the weighting
    - X (dict): A dictionary containing the original datasets
    - auxiliary_names (dict): A dictionary containing auxiliary naming
    - n_dim (int): The number of dimensions to be included in the resulting DataFrames.

    The function updates the 'analysis_result' dictionary with three new keys:
    - 'lambda': Eigenvalues or metrics of variance explained, indexed by 'tab_names' from 'multi_block_eigen_data'.
    - 'SynVar': Synthetic variables calculated from 'average_view_u', indexed by 'row.names' from 'X'.
    - 'axis': Axis projections derived from 'v_k', indexed by 'col' names from 'auxiliary_names'.
    """

    analysis_result['lambda'] = pd.DataFrame(lambda_matrix, index=multi_block_eigen_data['tab_names'],
                                             columns=[f'lam{i}' for i in range(1, n_dim + 1)])

    analysis_result['SynVar'] = pd.DataFrame(np.array(average_view_u).T[:, :n_dim], index=X['row.names'],
                                             columns=[f'SynVar{i}' for i in range(1, n_dim + 1)])

    analysis_result['axis'] = pd.DataFrame(np.array(v_k).T[:, :n_dim], index=auxiliary_names['col'],
                                           columns=[f'Axis{i}' for i in range(1, n_dim + 1)])


def multiple_coinertia_analysis(datasets, weight_option=None, n_dim=3, is_data_being_projected=False):
    """
    Performs a Multiple Co-Inertia Analysis (MCOA) on one or multiple datasets

    Parameters:
    - datasets (dict): A dictionary containing the datasets and associated metadata structured for MCOA
    - weight_option (list of str, optional): A list specifying the weighting scheme to be used. Default schemes include 'inertia', 'lambda1', 'uniform', and 'internal'.
    - n_dim (int, optional): The number of dimensions to project the datasets onto. Defaults to 3.
    - is_data_being_projected (bool, optional): Flag indicating whether the data is being projected. Defaults to False.

    Key steps:
    1. Validates the input datasets and weights them
    2. Calculates weights for rows and columns and pre-processes the datasets
    3. Conducts a joint SVD to determine principal components and eigenvalues
    4. Projects the datasets onto the identified principal axes, adjusting for row and column weights
    5. Calculates synthetic variables
    6. Aggregates the analysis results into a structured output, "analysis_result"

    Returns:
    - dict: An updated dictionary ('analysis_result') containing the results of the MCOA

    Raises:
    - ValueError: If the input datasets are not of the expected class 'ktab' or if an unknown weighting option is provided.

    """

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
        weight_table = [1 / multi_block_eigen_data["eigenvalues"][rank_per_block[i - 1]][0] for i in
                        range(1, number_Datasets + 1)]
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

    x_Tilde = pd.DataFrame(datasets[0])

    if is_data_being_projected:
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

        singular_value_list = np.concatenate(
            [singular_value_list, singular_value]) if singular_value_list is not None else singular_value

    n_dim = max(n_dim, 2)
    analysis_result = {'pseudo_eigenvalues': singular_value_list ** 2}
    rank_per_block_array = np.array(rank_per_block)
    lambda_matrix = np.zeros((number_Datasets, n_dim))

    for dataset_index in range(number_Datasets):
        is_current_dataset = (rank_per_block_array == dataset_index + 1)
        current_eigenvalues = multi_block_eigen_data['eigenvalues'][is_current_dataset]
        current_rank = min(multi_block_eigen_data['rank'][dataset_index], n_dim)
        lambda_matrix[dataset_index, :current_rank] = current_eigenvalues[:current_rank]

    create_analysis_dataframes(
        analysis_result=analysis_result,
        lambda_matrix=lambda_matrix,
        average_view_u=average_View_U,
        v_k=v_k_Normalized,
        multi_block_eigen_data=multi_block_eigen_data,
        X=datasets,
        auxiliary_names=auxiliary_names,
        n_dim=n_dim
    )

    calculate_row_projection(
        number_rows=number_rows,
        number_Datasets=number_Datasets,
        n_dim=n_dim,
        block_Indicator=block_Indicator,
        dataset_index=dataset_indix,
        analysis_result=analysis_result,
        datasets=datasets,
        column_weight=column_weight,
        row_weight=row_weight,
        auxiliary_names=auxiliary_names
    )

    row_normed_column_projection(
        number_rows=number_rows,
        number_Datasets=number_Datasets,
        n_dim=n_dim,
        row_weight=row_weight,
        datasets=datasets,
        auxiliary_names=auxiliary_names,
        analysis_result=analysis_result,
        number_Columns=number_Columns
    )

    analysis_result['nf'] = n_dim
    analysis_result['TL'] = datasets['TL']
    analysis_result['TC'] = datasets['TC']
    analysis_result['T4'] = datasets['T4']
    analysis_result['class'] = 'mcoa'

    return analysis_result
