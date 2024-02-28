import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralCoclustering

def perform_clustering(mcia_instance, k=3, n_clusters=8, threshold_percentiles=(3, 25, 1)):

    df_genes = mcia_instance.column_projection

    current_suffix = df_genes.index[0].split('.')[-1]
    alternative_suffix = "Ana1" if current_suffix == "Ana2" else "Ana2"

    if df_genes.index[0].startswith('chr'):
        df_genes_genes = df_genes[df_genes.index.str.endswith(alternative_suffix)]
        df_genes_peaks = df_genes[df_genes.index.str.endswith(current_suffix)]
    else:
        df_genes_peaks = df_genes[df_genes.index.str.endswith(alternative_suffix)]
        df_genes_genes = df_genes[df_genes.index.str.endswith(current_suffix)]

    df_genes_peaks_2 = df_genes_peaks.copy()
    df_genes_genes_2 = df_genes_genes.copy()
    df_genes_peaks_2['peak_index'] = [i + 0.1 for i in range(len(df_genes_peaks))]
    df_genes_genes_2['gene_index'] = [i + 0.2 for i in range(len(df_genes_genes))]

    gene_norms = df_genes_genes.apply(np.linalg.norm, axis=1)
    threshold_gene = np.percentile(gene_norms, threshold_percentiles[0])
    selected_genes = gene_norms[gene_norms > threshold_gene].index

    peak_norms = df_genes_peaks.apply(np.linalg.norm, axis=1)
    threshold_peak = np.percentile(peak_norms, threshold_percentiles[1])
    selected_peaks = peak_norms[peak_norms > threshold_peak].index

    df_genes_genes_filtered = df_genes_genes_2.loc[selected_genes]
    df_genes_peaks_filtered = df_genes_peaks_2.loc[selected_peaks]

    df_cells = mcia_instance.SynVar

    def calculate_knn(df_from, df_to, k_neighbors):
        cosine_distances = cdist(df_from, df_to, metric='cosine')
        return np.argsort(cosine_distances, axis=1)[:, :k_neighbors]

    knn_peaks = df_genes_peaks_filtered.drop(columns=['peak_index'])
    knn_genes = df_genes_genes_filtered.drop(columns=['gene_index'])

    knn_cells_to_genes = calculate_knn(df_cells, knn_genes, k)
    knn_cells_to_peaks = calculate_knn(df_cells, knn_peaks, k)

    subset_peak_index = df_genes_peaks_2.loc[knn_peaks.index, 'peak_index']
    subset_gene_index = df_genes_genes_2.loc[knn_genes.index, 'gene_index']


    n_cells = df_cells.shape[0]
    n_genes = df_genes_genes_filtered.shape[0]
    n_peaks = df_genes_peaks_filtered.shape[0]
    extended_adjacency_matrix = np.zeros((n_cells, n_genes + n_peaks))

    for i in range(n_cells):
        extended_adjacency_matrix[i, knn_cells_to_genes[i]] = 1
        extended_adjacency_matrix[i, n_genes + knn_cells_to_peaks[i]] = 1

    extended_adjacency_matrix_with_indices = np.zeros((n_cells + 1, n_genes + n_peaks))
    extended_adjacency_matrix_with_indices[:n_cells, :n_genes + n_peaks] = extended_adjacency_matrix
    extended_adjacency_matrix_with_indices[-1, :n_genes] = df_genes_genes_filtered['gene_index'].values
    extended_adjacency_matrix_with_indices[-1, n_genes:n_genes + n_peaks] = df_genes_peaks_filtered['peak_index'].values

    sum_per_column = np.sum(extended_adjacency_matrix_with_indices[:-1, :], axis=0)
    columns_with_expression = sum_per_column > np.percentile(sum_per_column, threshold_percentiles[2])
    matrix_without_id_row = extended_adjacency_matrix_with_indices[:-1, columns_with_expression]

    column_names = extended_adjacency_matrix_with_indices[-1, columns_with_expression]
    final_dataframe = pd.DataFrame(matrix_without_id_row, columns=column_names)

    geni = [col for col in final_dataframe.columns if str(col).endswith('.2')]
    picchi = [col for col in final_dataframe.columns if str(col).endswith('.1')]
    geni = [str(col).split('.')[0] for col in geni]
    picchi = [str(col).split('.')[0] for col in picchi]

    model = SpectralCoclustering(n_clusters=n_clusters)
    model.fit(matrix_without_id_row)

    row_labels = model.row_labels_
    column_labels = model.column_labels_

    return extended_adjacency_matrix, model, row_labels, column_labels

