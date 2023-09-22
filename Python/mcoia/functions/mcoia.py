import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from .sepan import multi_block_eigenanalysis
from .mcoa_processing import ktab_util_names, normalize_matrix_by_block, recalculate

def multiple_coinertia_analysis(X, option=None, nf=3, data_projected = False):
    if option is None:
        option = ["inertia", "lambda1", "uniform", "internal"]
    if X.get('class') != "ktab":
        raise ValueError("Expected object of class 'ktab'")
    option = option[0]
    if option == "internal":
        if X.get('weight_table') is None:
            print("Internal weights not found: uniform weights are used")
            option = "uniform"

    lw = X['row_weight']
    nlig = len(lw)
    cw = X['column_weight']
    ncol = len(cw)
    nbloc = len(X['blocks'])
    indicablo = X['TC']['T']
    veclev = sorted(list(set(X['TC']['T'])))
    #  todo: somehow this (the sorted argument) fixes the problem that sometimes the analysis order are swapped,
    #  causing wrong results. This is present somehow only in tests, but it seems like it's not a problem for a
    #  normal call of the functions. Will try to dive deeper into it after
    multi_block_eigen_data = multi_block_eigenanalysis(X, nf=4)  # This is used to calculate the component scores factor scores for each data

    rank_fac = list(np.repeat(range(1, nbloc + 1), multi_block_eigen_data["rank"]))

    auxiliary_names = ktab_util_names(X)
    sums = {}

    if option == "lambda1":
        weight_table = [1 / multi_block_eigen_data["eigenvalues"][rank_fac[i - 1]][0] for i in range(1, nbloc + 1)]
    elif option == "inertia":
        # Iterate over rank_fac and multi_block_eigen_data['Eig'] simultaneously
        for rank, value in zip(rank_fac, multi_block_eigen_data['eigenvalues']):
            # If rank is not in sums, initialize with value, otherwise accumulate
            sums[rank] = sums.get(rank, 0) + value

        # Create weight_table by taking reciprocals of the accumulated sums
        weight_table = [1 / sums[i] for i in sorted(sums.keys())]  # This is done to assign weights to each rank
    elif option == "uniform":
        weight_table = [1] * nbloc
    elif option == "internal":
        weight_table = X['weight_table']
    else:
        raise ValueError("Unknown option")

    for i in range(nbloc):
        X[i] = [X[i] * np.sqrt(weight_table[i])]  # We are weighting the datasets according to the calculated eigenvalues

    for k in range(nbloc):
        X[k] = pd.DataFrame(X[k][0])

    multi_block_eigen_data = multi_block_eigenanalysis(X, nf=4)  # Recalculate sepan with the updated X

    # Convert the first element of X to a DataFrame and assign it to complete_weighted_datasets
    complete_weighted_datasets = pd.DataFrame(X[0])

    if data_projected:
        return X[0]

    # Concatenate the remaining elements of X (from the 2nd to nbloc) to the columns of complete_weighted_datasets
    for i in range(1, nbloc):
        complete_weighted_datasets = pd.concat([complete_weighted_datasets, pd.DataFrame(X[i])], axis=1)

    '''
    This creates the merged table of K weighted datasets
    '''

    # Assign the names of the columns of complete_weighted_datasets from auxiliary_names['col']
    complete_weighted_datasets.columns = auxiliary_names['col']

    complete_weighted_datasets = complete_weighted_datasets.mul(np.sqrt(lw), axis=0)
    complete_weighted_datasets = complete_weighted_datasets.mul(np.sqrt(cw), axis=1)

    compogene = []
    u_k_normalized = []
    singular_value_list = None

    nfprovi = min(20, nlig, ncol)
    # Perform SVD computations
    for i in range(nfprovi):

        truncated_svd = TruncatedSVD(n_components=nfprovi)
        u = truncated_svd.fit_transform(complete_weighted_datasets)
        s = truncated_svd.singular_values_
        vt = truncated_svd.components_
        u = u / s

        # Extract the first column of u and normalize by the square root of lw (row_weights)
        normalized_u = u[:, 0] / np.sqrt(lw)
        compogene.append(normalized_u)

        # Extract the first column of vt (v transposed in SVD), then normalize it
        normalized_v = normalize_matrix_by_block(vt[0, :], nbloc, indicablo, veclev)

        # Re-calculate complete_weighted_datasets
        complete_weighted_datasets = recalculate(complete_weighted_datasets, normalized_v, nbloc, indicablo, veclev)

        normalized_v /= np.sqrt(cw)
        u_k_normalized.append(normalized_v)

        # Extract the first singular value
        singular_value = np.array([s[0]])

        singular_value_list = np.concatenate([singular_value_list, singular_value]) if singular_value_list is not None else singular_value
    # Squaring the singular_value_list to get pseudo eigenvalues
    pseudo_eigenvalues = singular_value_list ** 2

    if nf <= 0:
        nf = 2

    analysis_result = {'pseudo_eigenvalues': pseudo_eigenvalues}
    rank_fac = np.array(rank_fac)
    lambda_matrix = np.zeros((nbloc, nf))
    for i in range(1, nbloc + 1):
        mask = (rank_fac == i)

        # Filter out the eigenvalues using the mask
        w1 = multi_block_eigen_data['eigenvalues'][mask]

        r0 = multi_block_eigen_data['rank'][i - 1]
        if r0 > nf:
            r0 = nf

        # Assign values to the lambda_matrix
        lambda_matrix[i - 1, :r0] = w1[:r0]

    # Convert the matrix to a DataFrame and assign row and column names
    lambda_df = pd.DataFrame(lambda_matrix)
    lambda_df.index = multi_block_eigen_data['tab_names']
    lambda_df.columns = [f'lam{i}' for i in range(1, nf + 1)]
    analysis_result['lambda'] = lambda_df

    # Create a DataFrame for synthesized variables
    syn_var_df = pd.DataFrame(np.array(compogene).T[:, :nf])  # should contain the u_k
    syn_var_df.columns = [f'SynVar{i}' for i in range(1, nf + 1)]
    syn_var_df.index = X['row.names']
    analysis_result['SynVar'] = syn_var_df

    # Create a DataFrame for axes
    axis_df = pd.DataFrame(np.array(u_k_normalized).T[:, :nf])  # u_k_normalized should be the v_k
    axis_df.columns = [f'Axis{i}' for i in range(1, nf + 1)]
    axis_df.index = auxiliary_names['col']
    analysis_result['axis'] = axis_df

    # `w` is a matrix to store the transformed data for each data point across all blocks.
    w = np.zeros((nlig * nbloc, nf))

    # `covar` will store covariance-like values for each block.
    covar = np.zeros((nbloc, nf))
    i2 = 0
    current_index = 0  # Pointer for rows in `w`.

    # Start iterating over blocks.
    for k in range(nbloc):
        # Update the slice pointers.
        i2 = i2 + nlig

        # Create a mask for extracting data of the current block.
        mask = indicablo == veclev[k]

        vk = analysis_result['axis'].reset_index(drop=True).loc[mask].values
        complete_weighted_datasets = np.array(X[k])

        cw_array = np.array(cw)
        vk *= cw_array[mask].reshape(-1, 1)

        projection = complete_weighted_datasets @ vk

        # Populate the `w` matrix with the computed values for the current block.
        rows_in_projection = projection.shape[0]
        w[current_index:current_index + rows_in_projection, :] = projection
        current_index += rows_in_projection

        scaled_data = (projection * analysis_result['SynVar'].values) * lw.reshape(-1, 1)

        covar[k, :] = scaled_data.sum(axis=0)

    # Convert w to DataFrame with appropriate row names and column names
    w_df = pd.DataFrame(w, index=auxiliary_names['row'])
    w_df.columns = [f"Axis{str(i + 1)}" for i in range(nf)]

    analysis_result['Tli'] = w_df

    # Convert covar to DataFrame and square it, then store in analysis_result
    covar_df = pd.DataFrame(covar)
    covar_df.index = X['tab.names']
    covar_df.columns = [f"cov2{str(i + 1)}" for i in range(nf)]
    analysis_result['cov2'] = covar_df ** 2

    # Initialize w and indices
    w = np.zeros((nlig * nbloc, nf))
    i2 = 0
    # Iterate over blocks to adjust w based on Tli and sqrt of lw
    for k in range(nbloc):
        i1 = i2 + 1
        i2 = i2 + nlig
        complete_weighted_datasets = analysis_result['Tli'].iloc[i1 - 1:i2, :]
        lw_sqrt = np.sqrt(lw)
        squared_values = (complete_weighted_datasets.values * lw_sqrt.reshape(-1, 1)) ** 2
        column_sums_sqrt = np.sqrt(squared_values.sum(axis=0))
        complete_weighted_datasets = complete_weighted_datasets.divide(column_sums_sqrt)
        w[i1 - 1:i2, :] = complete_weighted_datasets.values

    # Create DataFrame for adjusted w and store it as Tl1 in analysis_result
    w_df = pd.DataFrame(w, index=auxiliary_names['row'])
    w_df.columns = [f"Axis{str(i + 1)}" for i in range(nf)]
    analysis_result['Tl1'] = w_df  # a normalized and re-scaled version of Tli

    w = np.zeros((ncol, nf))
    i2 = 0
    # Iterate over blocks to update w based on SynVar and lw
    for k in range(nbloc):
        i1 = i2 + 1
        i2 = i2 + X[k].shape[1]
        urk = np.array(analysis_result['SynVar'])
        complete_weighted_datasets = np.array(X[k])
        urk = urk * lw[:, np.newaxis]
        w[i1 - 1:i2, :] = complete_weighted_datasets.T.dot(urk)

    # Create DataFrame for w and store it as Tco in analysis_result
    w_df = pd.DataFrame(w, index=auxiliary_names['col'])
    w_df.columns = [f"SV{str(i + 1)}" for i in range(nf)]
    analysis_result['Tco'] = w_df

    # Reset variables and initialize var.names
    var_names = []
    w = np.zeros((nbloc * 4, nf))
    i2 = 0
    block_indices = indicablo.reset_index(drop=True)

    # Iterate over blocks to update w and var.names based on axis and multi_block_eigen_data
    for k in range(nbloc):
        i1 = i2 + 1
        i2 = i2 + 4

        bool_filter = block_indices == veclev[k]

        urk = analysis_result['axis'].reset_index(drop=True)[bool_filter].values
        complete_weighted_datasets = multi_block_eigen_data['column_scores'].reset_index(drop=True)[bool_filter].values
        bool_filter_array = np.array(bool_filter)
        filtered_cw = np.array([cw[i] for i, flag in enumerate(bool_filter_array) if flag]).reshape(-1, 1)
        urk = urk * filtered_cw
        complete_weighted_datasets = complete_weighted_datasets.T.dot(urk)
        complete_weighted_datasets = np.nan_to_num(
            complete_weighted_datasets)

        for i in range(min(nf, 4)):
            if complete_weighted_datasets[i, i] < 0:
                complete_weighted_datasets[i, :] = -complete_weighted_datasets[i, :]

        w[i1 - 1:i2, :] = complete_weighted_datasets
        var_names.extend([f"{multi_block_eigen_data['tab_names'][k]}.a{str(i + 1)}" for i in range(4)])

    w_df = pd.DataFrame(w, index=auxiliary_names['tab'])
    w_df.columns = [f"Axis{str(i + 1)}" for i in range(nf)]
    analysis_result['Tax'] = w_df

    # Set additional properties of analysis_result
    analysis_result['nf'] = nf
    analysis_result['TL'] = X['TL']
    analysis_result['TC'] = X['TC']
    analysis_result['T4'] = X['T4']
    analysis_result['class'] = 'mcoa'

    return analysis_result
