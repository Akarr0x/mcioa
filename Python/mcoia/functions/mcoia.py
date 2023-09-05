import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from .sepan import sepan
from .mcoa_processing import ktab_util_names, normalize_per_block, recalculate

def mcoa(X, option=None, nf=3, data_projected = False):
    if option is None:
        option = ["inertia", "lambda1", "uniform", "internal"]
    if X.get('class') != "ktab":
        raise ValueError("Expected object of class 'ktab'")
    option = option[0]
    if option == "internal":
        if X.get('tabw') is None:
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
    Xsepan = sepan(X, nf=4)  # This is used to calculate the component scores factor scores for each data

    rank_fac = list(np.repeat(range(1, nbloc + 1), Xsepan["rank"]))

    auxinames = ktab_util_names(X)
    sums = {}

    if option == "lambda1":
        tabw = [1 / Xsepan["eigenvalues"][rank_fac[i - 1]][0] for i in range(1, nbloc + 1)]
    elif option == "inertia":
        # Iterate over rank_fac and Xsepan['Eig'] simultaneously
        for rank, value in zip(rank_fac, Xsepan['eigenvalues']):
            # If rank is not in sums, initialize with value, otherwise accumulate
            sums[rank] = sums.get(rank, 0) + value

        # Create tabw by taking reciprocals of the accumulated sums
        tabw = [1 / sums[i] for i in sorted(sums.keys())]  # This is done to assign weights to each rank
    elif option == "uniform":
        tabw = [1] * nbloc
    elif option == "internal":
        tabw = X['tabw']
    else:
        raise ValueError("Unknown option")

    for i in range(nbloc):
        X[i] = [X[i] * np.sqrt(tabw[i])]  # We are weighting the datasets according to the calculated eigenvalues

    for k in range(nbloc):
        X[k] = pd.DataFrame(X[k][0])

    Xsepan = sepan(X, nf=4)  # Recalculate sepan with the updated X


    '''
    The call of two sepan functions, one with the non weighted dataset and the other with the weighted tables 
    is done so that the contributions of each datasets are balanced, so that the co-inertia structure better reflects
    shared patterns of the different datasets, not just patterns from the most variable dataset.
    Basically this is calculating \( X^† = [w^{\frac{1}{2}}_1 X_1, w^{\frac{1}{2}}_2 X_2, \dots, w^{\frac{1}{2}}_K X_K]\

    '''

    # Convert the first element of X to a DataFrame and assign it to tab
    tab = pd.DataFrame(X[0])

    if data_projected:
        return X[0]

    # Concatenate the remaining elements of X (from the 2nd to nbloc) to the columns of tab
    for i in range(1, nbloc):
        tab = pd.concat([tab, pd.DataFrame(X[i])], axis=1)

    '''
    This creates the merged table of K weighted datasets
    '''

    # Assign the names of the columns of tab from auxinames['col']
    tab.columns = auxinames['col']

    '''
    This is used to perform combined row and column weighting transformations to the data.
    The row weights, represented by \(D^{1/2}\), are multiplied with the data to adjust the influence 
    of each row (sample) in the final analysis. 
    This ensures that each row's contributions are scaled according to its importance or frequency.

    The column weights, represented by \(Q^{1/2}_k\), are multiplied with the data to adjust the influence 
    of each column (feature) in the corresponding data block \(X_k\). 
    This accounts for variable importance, scaling, or measurement units, ensuring that 
    each feature's contribution is balanced when combined with other datasets.

    These operations transform each data block \(X_k\) into \(\tilde{X}_k = w_k^{1/2} D^{1/2} X_k Q_k^{1/2}\),
    as per the This results in a weighted dataset that is used in the subsequent optimization problem.
    '''

    tab = tab.mul(np.sqrt(lw), axis=0)
    tab = tab.mul(np.sqrt(cw), axis=1)


    '''
    Initialization for upcoming calculations.
    compogene and uknorme are lists that will hold computation results, 
    while valsing might be for singular values, but we would need more context to be certain.
    '''
    compogene = []
    uknorme = []
    valsing = None

    '''
    Determine how many SVD iterations or components to compute,
    limiting the number of singular vectors/values to a maximum of 20.
    '''

    nfprovi = min(20, nlig, ncol)
    # Perform SVD computations
    for i in range(nfprovi):
        '''
        Compute the Singular Value Decomposition (SVD) of the weighted matrix tab.
        Mathematically, given a matrix A (tab), the SVD is represented as:
        \( A = U \Sigma V^T \)
        where U and V are orthogonal matrices and \(\Sigma\) is a diagonal matrix with singular values.
        '''

        truncated_svd = TruncatedSVD(n_components=nfprovi)
        u = truncated_svd.fit_transform(tab)
        s = truncated_svd.singular_values_
        vt = truncated_svd.components_
        u = u / s

        '''
        We extract the first column of U and the first row of V^T for a couple of key reasons:

        1. The first column of U (u[:, 0]) captures the most "effective" linear combination of rows 
           (e.g., genes in a gene expression dataset) in representing the original data.

        2. Similarly, the first row of V^T (vt[0, :]) captures the most "effective" linear combination of columns 
           (e.g., samples or cells in a gene expression dataset) in representing the original data.

        In essence, these vectors are the principal axes that maximize the variance in the data and are used 
        for the projections.
        '''
        # Extract the first column of u and normalize by the square root of lw (row_weights)
        normalized_u = u[:, 0] / np.sqrt(lw)
        compogene.append(normalized_u)  # Append to the list of compogene

        # Extract the first column of vt (v transposed in SVD), then normalize it
        normalized_v = normalize_per_block(vt[0, :], nbloc, indicablo, veclev)

        '''
        This deflationary step recalculates 'tab' to remove the influence of the previously computed 
        synthetic center 'normalized_v'. By doing this, we're setting the stage to find 
        the next orthogonal synthetic center in subsequent iterations. 
        The method ensures that the influence of each synthetic center is considered only once, 
        and in each subsequent iteration, 
        the "most influential" direction in the residual (or deflated) data is sought after.
        '''

        # Re-calculate tab
        tab = recalculate(tab, normalized_v, nbloc, indicablo, veclev)

        normalized_v /= np.sqrt(cw)
        uknorme.append(normalized_v)

        # Extract the first singular value
        singular_value = np.array([s[0]])

        valsing = np.concatenate([valsing, singular_value]) if valsing is not None else singular_value
    # Squaring the valsing to get pseudo eigenvalues
    pseudo_eigenvalues = valsing ** 2



    # Ensure nf is at least 2
    if nf <= 0:
        nf = 2

    acom = {'pseudo_eigenvalues': pseudo_eigenvalues}
    rank_fac = np.array(rank_fac)
    lambda_matrix = np.zeros((nbloc, nf))
    for i in range(1, nbloc + 1):
        mask = (rank_fac == i)

        # Filter out the eigenvalues using the mask
        w1 = Xsepan['eigenvalues'][mask]

        r0 = Xsepan['rank'][i - 1]
        if r0 > nf:
            r0 = nf

        # Assign values to the lambda_matrix
        lambda_matrix[i - 1, :r0] = w1[:r0]

    # Convert the matrix to a DataFrame and assign row and column names
    lambda_df = pd.DataFrame(lambda_matrix)
    lambda_df.index = Xsepan['tab_names']
    lambda_df.columns = [f'lam{i}' for i in range(1, nf + 1)]
    acom['lambda'] = lambda_df

    # Create a DataFrame for synthesized variables
    syn_var_df = pd.DataFrame(np.array(compogene).T[:, :nf])  # should contain the u_k
    syn_var_df.columns = [f'SynVar{i}' for i in range(1, nf + 1)]
    syn_var_df.index = X['row.names']
    acom['SynVar'] = syn_var_df

    # Create a DataFrame for axes
    axis_df = pd.DataFrame(np.array(uknorme).T[:, :nf])  # uknorme should be the v_k
    axis_df.columns = [f'Axis{i}' for i in range(1, nf + 1)]
    axis_df.index = auxinames['col']
    acom['axis'] = axis_df

    # Initialize matrices for further processing:

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

        '''
        Fetch the computed v_k values for the current block. 
        Mathematically:
        v_k = the kth vector from the right singular matrix corresponding to the kth block.
        After the weight multiplication we can see that this is equal to  QV
        '''
        vk = acom['axis'].reset_index(drop=True).loc[mask].values
        tab = np.array(X[k])

        cw_array = np.array(cw)
        vk *= cw_array[mask].reshape(-1, 1)

        '''
        Compute the product of the data matrix for the block and the weighted v_k values.
        This operation gives a projection of the data onto the direction of v_k.
        Mathematically:
        projection_k = X_k * QV
        '''
        projection = tab @ vk

        # Populate the `w` matrix with the computed values for the current block.
        rows_in_projection = projection.shape[0]
        w[current_index:current_index + rows_in_projection, :] = projection
        current_index += rows_in_projection

        '''
        Scale the projection values using the synthetic variables (u_k values).
        This operation ensures that the resulting values are in terms of how much they align 
        with the primary singular vectors. Then, it's scaled further using row weights.
        Mathematically:
        scaled_data = (projection_k * u) * D.
        '''
        scaled_data = (projection * acom['SynVar'].values) * lw.reshape(-1, 1)

        '''
        Sum the processed data for the current block. This step computes a cumulative 
        metric
        covar_k = sum(scaled_data)
        '''
        covar[k, :] = scaled_data.sum(axis=0)

    # Convert w to DataFrame with appropriate row names and column names
    w_df = pd.DataFrame(w, index=auxinames['row'])
    w_df.columns = [f"Axis{str(i + 1)}" for i in range(nf)]
    '''
    w is defined as X_k * VQ
    '''
    acom['Tli'] = w_df

    # Convert covar to DataFrame and square it, then store in acom
    covar_df = pd.DataFrame(covar)
    covar_df.index = X['tab.names']  # todo: check value
    covar_df.columns = [f"cov2{str(i + 1)}" for i in range(nf)]
    acom['cov2'] = covar_df ** 2

    # Initialize w and indices
    w = np.zeros((nlig * nbloc, nf))
    i2 = 0
    # Iterate over blocks to adjust w based on Tli and sqrt of lw
    for k in range(nbloc):
        i1 = i2 + 1
        i2 = i2 + nlig
        tab = acom['Tli'].iloc[i1 - 1:i2, :]
        lw_sqrt = np.sqrt(lw)
        squared_values = (tab.values * lw_sqrt.reshape(-1, 1)) ** 2
        column_sums_sqrt = np.sqrt(squared_values.sum(axis=0))
        tab = tab.divide(column_sums_sqrt)
        w[i1 - 1:i2, :] = tab.values

    # Create DataFrame for adjusted w and store it as Tl1 in acom
    w_df = pd.DataFrame(w, index=auxinames['row'])
    w_df.columns = [f"Axis{str(i + 1)}" for i in range(nf)]
    acom['Tl1'] = w_df  # a normalized and re-scaled version of Tli

    w = np.zeros((ncol, nf))
    i2 = 0
    # Iterate over blocks to update w based on SynVar and lw
    for k in range(nbloc):
        i1 = i2 + 1
        i2 = i2 + X[k].shape[1]
        urk = np.array(acom['SynVar'])
        tab = np.array(X[k])
        urk = urk * lw[:, np.newaxis]
        w[i1 - 1:i2, :] = tab.T.dot(urk)

    # Create DataFrame for w and store it as Tco in acom
    w_df = pd.DataFrame(w, index=auxinames['col'])
    w_df.columns = [f"SV{str(i + 1)}" for i in range(nf)]
    acom['Tco'] = w_df

    # Reset variables and initialize var.names
    var_names = []
    w = np.zeros((nbloc * 4, nf))
    i2 = 0
    block_indices = indicablo.reset_index(drop=True)

    # Iterate over blocks to update w and var.names based on axis and Xsepan
    for k in range(nbloc):
        i1 = i2 + 1
        i2 = i2 + 4

        bool_filter = block_indices == veclev[k]

        urk = acom['axis'].reset_index(drop=True)[bool_filter].values
        tab = Xsepan['component_scores'].reset_index(drop=True)[bool_filter].values
        bool_filter_array = np.array(bool_filter)
        filtered_cw = np.array([cw[i] for i, flag in enumerate(bool_filter_array) if flag]).reshape(-1, 1)
        urk = urk * filtered_cw
        tab = tab.T.dot(urk)
        tab = np.nan_to_num(
            tab)  # todo Check that this does not give error. The result for two datasets works, but could be a hard
        # coding. The concept is that this function produced NaN instead of 0.00000, must be checked.

        for i in range(min(nf, 4)):
            if tab[i, i] < 0:
                tab[i, :] = -tab[i, :]

        w[i1 - 1:i2, :] = tab
        var_names.extend([f"{Xsepan['tab_names'][k]}.a{str(i + 1)}" for i in range(4)])

    w_df = pd.DataFrame(w, index=auxinames['tab'])
    w_df.columns = [f"Axis{str(i + 1)}" for i in range(nf)]
    acom['Tax'] = w_df

    # Set additional properties of acom
    acom['nf'] = nf
    acom['TL'] = X['TL']
    acom['TC'] = X['TC']
    acom['T4'] = X['T4']
    acom['class'] = 'mcoa'

    return acom
