import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralCoclustering
import re


def standardize_chromosome_names(df, chromosome_column='chromosome'):
    """
    Standardizes chromosome names to have 'chr' prefix in lowercase.

    Parameters:
    - df (pd.DataFrame): DataFrame containing chromosome information.
    - chromosome_column (str): Name of the column with chromosome identifiers.

    Returns:
    - pd.DataFrame: DataFrame with standardized chromosome names.
    """

    def standardize_prefix(name):
        name = name.strip()
        # Regex to capture possible prefixes
        match = re.match(r'^(?i)(chr|chromosome|chrom)?(?:osome)?(?:\s*)?([0-9XYM]+)$', name)
        if match:
            prefix, chrom = match.groups()
            return f'chr{chrom.upper()}'
        else:
            # If no match, return the name as is
            return name

    df[chromosome_column] = df[chromosome_column].apply(standardize_prefix)
    return df


def perform_clustering(mcia_instance, k=3, n_clusters=8, threshold_percentiles=(3, 25, 1)):
    """
    Performs clustering on the provided MCIA instance data.

    Parameters:
    - mcia_instance: An instance containing relevant genomic data.
    - k (int): Number of nearest neighbors.
    - n_clusters (int): Number of clusters for coclustering.
    - threshold_percentiles (tuple): Percentiles for filtering thresholds.

    Returns:
    - extended_adjacency_matrix (np.ndarray): The adjacency matrix used.
    - model (SpectralCoclustering): The fitted clustering model.
    - row_labels (np.ndarray): Row labels from clustering.
    - column_labels (np.ndarray): Column labels from clustering.
    """
    df_genes = mcia_instance.column_projection.copy()

    # Standardize chromosome names
    df_genes = standardize_chromosome_names(df_genes)

    # Extract the suffix from the first index
    first_index = df_genes.index[0]
    try:
        current_suffix = first_index.split('.')[-1]
    except IndexError:
        raise ValueError("Index format is unexpected. Ensure it contains at least one '.' character.")

    alternative_suffix = "Ana1" if current_suffix == "Ana2" else "Ana2"

    # Determine if chromosome names start with 'chr'
    has_chr_prefix = df_genes.index.str.match(r'(?i)^chr')

    if has_chr_prefix.iloc[0]:
        df_genes_peaks = df_genes[df_genes.index.str.endswith(current_suffix)]
        df_genes_genes = df_genes[df_genes.index.str.endswith(alternative_suffix)]
    else:
        df_genes_peaks = df_genes[df_genes.index.str.endswith(alternative_suffix)]
        df_genes_genes = df_genes[df_genes.index.str.endswith(current_suffix)]

    df_genes_peaks_2 = df_genes_peaks.copy()
    df_genes_genes_2 = df_genes_genes.copy()
    df_genes_peaks_2['peak_index'] = np.arange(len(df_genes_peaks)) + 0.1
    df_genes_genes_2['gene_index'] = np.arange(len(df_genes_genes)) + 0.2

    # Normalize and filter genes
    gene_norms = df_genes_genes.apply(np.linalg.norm, axis=1)
    threshold_gene = np.percentile(gene_norms, threshold_percentiles[0])
    selected_genes = gene_norms[gene_norms > threshold_gene].index

    # Normalize and filter peaks
    peak_norms = df_genes_peaks.apply(np.linalg.norm, axis=1)
    threshold_peak = np.percentile(peak_norms, threshold_percentiles[1])
    selected_peaks = peak_norms[peak_norms > threshold_peak].index

    df_genes_genes_filtered = df_genes_genes_2.loc[selected_genes]
    df_genes_peaks_filtered = df_genes_peaks_2.loc[selected_peaks]

    df_cells = mcia_instance.SynVar.copy()

    def calculate_knn(df_from, df_to, k_neighbors):
        """
        Calculates the k-nearest neighbors based on cosine distance.

        Parameters:
        - df_from (pd.DataFrame): Source dataframe.
        - df_to (pd.DataFrame): Target dataframe.
        - k_neighbors (int): Number of neighbors.

        Returns:
        - np.ndarray: Indices of nearest neighbors.
        """
        cosine_distances = cdist(df_from, df_to, metric='cosine')
        return np.argsort(cosine_distances, axis=1)[:, :k_neighbors]

    knn_peaks = df_genes_peaks_filtered.drop(columns=['peak_index'])
    knn_genes = df_genes_genes_filtered.drop(columns=['gene_index'])

    knn_cells_to_genes = calculate_knn(df_cells, knn_genes, k)
    knn_cells_to_peaks = calculate_knn(df_cells, knn_peaks, k)

    # Extract subset indices
    subset_peak_index = df_genes_peaks_2.loc[knn_peaks.index, 'peak_index'].values
    subset_gene_index = df_genes_genes_2.loc[knn_genes.index, 'gene_index'].values

    n_cells = df_cells.shape[0]
    n_genes = df_genes_genes_filtered.shape[0]
    n_peaks = df_genes_peaks_filtered.shape[0]

    # Initialize the adjacency matrix
    extended_adjacency_matrix = np.zeros((n_cells, n_genes + n_peaks))

    for i in range(n_cells):
        # Assign connections to genes
        extended_adjacency_matrix[i, knn_cells_to_genes[i]] = 1
        # Assign connections to peaks with offset
        extended_adjacency_matrix[i, n_genes + knn_cells_to_peaks[i]] = 1

    # Initialize matrix with an additional row for indices
    extended_adjacency_matrix_with_indices = np.zeros((n_cells + 1, n_genes + n_peaks))
    extended_adjacency_matrix_with_indices[:n_cells, :n_genes + n_peaks] = extended_adjacency_matrix
    extended_adjacency_matrix_with_indices[-1, :n_genes] = df_genes_genes_filtered['gene_index'].values
    extended_adjacency_matrix_with_indices[-1, n_genes:n_genes + n_peaks] = df_genes_peaks_filtered['peak_index'].values

    # Filter columns based on expression thresholds
    sum_per_column = np.sum(extended_adjacency_matrix_with_indices[:-1, :], axis=0)
    threshold_expression = np.percentile(sum_per_column, threshold_percentiles[2])
    columns_with_expression = sum_per_column > threshold_expression
    matrix_without_id_row = extended_adjacency_matrix_with_indices[:-1, columns_with_expression]

    # Extract column names
    column_names = extended_adjacency_matrix_with_indices[-1, columns_with_expression]
    final_dataframe = pd.DataFrame(matrix_without_id_row, columns=column_names)

    # Separate genes and peaks based on suffix
    genes = [col for col in final_dataframe.columns if str(col).endswith('.2')]
    peaks = [col for col in final_dataframe.columns if str(col).endswith('.1')]
    genes = [str(col).split('.')[0] for col in genes]
    peaks = [str(col).split('.')[0] for col in peaks]

    # Perform spectral coclustering
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
    model.fit(matrix_without_id_row)

    row_labels = model.row_labels_
    column_labels = model.column_labels_

    return extended_adjacency_matrix, model, row_labels, column_labels

