import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import muon as mu

import mudatasets
mdata = mudatasets.load("brain3k_multiome", full=True)
mdata.var_names_make_unique()

rna = mdata['rna']

rna.var['mt'] = rna.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc.pl.violin(rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)


print(f"Before: {rna.n_obs} cells")
mu.pp.filter_obs(rna, 'n_genes_by_counts', lambda x: (x >= 200) & (x < 8000))
print(f"(After n_genes: {rna.n_obs} cells)")
mu.pp.filter_obs(rna, 'total_counts', lambda x: x < 40000)
print(f"(After total_counts: {rna.n_obs} cells)")
mu.pp.filter_obs(rna, 'pct_counts_mt', lambda x: x < 2)
print(f"After: {rna.n_obs} cells")


sc.pl.violin(rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)


rna.layers["counts"] = rna.X.copy()
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
# rna.raw = rna
rna.layers["lognorm"] = rna.X.copy()

sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
sc.pl.highly_variable_genes(rna)
sc.pp.scale(rna, max_value=10)
sc.tl.pca(rna, svd_solver='arpack')
sc.pl.pca(rna, color=['NRCAM', 'SLC1A2', 'SRGN', 'VCAN'])


from muon import atac as ac

atac = mdata.mod['atac']
sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
mu.pl.histogram(atac, ['n_genes_by_counts', 'total_counts'], linewidth=0)
mu.pp.filter_var(atac, 'n_cells_by_counts', lambda x: x >= 10)
print(f"Before: {atac.n_obs} cells")
mu.pp.filter_obs(atac, 'total_counts', lambda x: (x >= 1000) & (x <= 80000))
print(f"(After total_counts: {atac.n_obs} cells)")
mu.pp.filter_obs(atac, 'n_genes_by_counts', lambda x: (x >= 100) & (x <= 30000))
print(f"After: {atac.n_obs} cells")
mu.pl.histogram(atac, ['n_genes_by_counts', 'total_counts'], linewidth=0)
ac.pl.fragment_histogram(atac, region='chr1:1-2000000')
ac.tl.nucleosome_signal(atac, n=1e6)
mu.pl.histogram(atac, "nucleosome_signal", linewidth=0)
ac.tl.get_gene_annotation_from_rna(mdata['rna']).head(3)
tss = ac.tl.tss_enrichment(mdata, n_tss=1000)
ac.pl.tss_enrichment(tss)



atac.layers["counts"] = atac.X.copy()
sc.pp.normalize_total(atac, target_sum=1e4)
sc.pp.log1p(atac)
atac.layers["lognorm"] = atac.X.copy()
sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5)
sc.pl.highly_variable_genes(atac)
sc.pp.scale(atac, max_value=10)
sc.tl.pca(atac, svd_solver='arpack')
sc.pp.neighbors(atac, n_neighbors=10, n_pcs=20)
sc.tl.leiden(atac, resolution=.5)

ac.tl.rank_peaks_groups(atac, 'leiden', method='t-test')
result = atac.uns['rank_genes_groups']
groups = result['names'].dtype.names

try:
    pd.set_option("max_columns", 50)
except:
    pd.set_option("display.max_columns", 50)

pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'genes', 'pvals']}).head(10)


new_index_dict = {}
peak_sel = atac.uns["atac"]["peak_annotation"]
peak_sel = peak_sel[peak_sel.peak.isin(atac.var_names.values)]
gene_names = peak_sel.index
for gene_name, row in peak_sel.iterrows():
    peak = row['peak']
    matching_indices = atac.var[atac.var['gene_ids'] == peak].index
    for idx in matching_indices:
        new_index_dict[idx] = gene_name

atac.var.index = [new_index_dict.get(idx, idx) for idx in atac.var.index]


################################
import anndata
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import muon as mu

import mudatasets
a = anndata.read_h5ad("/home/diamanti/Thesis/10X_pbmc_atac_seurat_obj/scMultiome_seurat_pbmc.atac.h5ad")
import csv
cell_names_list = []
cell_name = pd.DataFrame()
with open("/home/diamanti/Thesis//10X_pbmc_atac_seurat_obj/scMultiome_seurat_pbmc.atac_activity_cellIDS.csv", mode='r') as file:
    csv_dict_reader = csv.DictReader(file)
    for row in csv_dict_reader:
        cell_names_list.append(pd.Series(1, name=row['x']))
cell_name = pd.concat(cell_names_list, axis=1).T

print(cell_name)

gene_name_list = []
gene_name = pd.DataFrame()
with open("/home/diamanti/Thesis/10X_pbmc_atac_seurat_obj/scMultiome_seurat_pbmc.atac_activity_genes.csv", mode='r') as file:
    csv_dict_reader = csv.DictReader(file)
    for row in csv_dict_reader:
        gene_name_list.append(pd.Series(1, name=row['x']))
gene_name = pd.concat(gene_name_list, axis=1).T

print(gene_name)

atac_to_expression_dataframe = a.X
atac_to_expression_dataframe = pd.DataFrame(atac_to_expression_dataframe.toarray())
atac_to_expression_dataframe.columns = gene_name.index
atac_to_expression_dataframe.index = cell_name.index


import os

data_dir = "/home/diamanti/Thesis"
mdata = mu.read_10x_h5(os.path.join(data_dir, "/home/diamanti/Thesis/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix-2.h5"))
mdata.var_names_make_unique()

rna = mdata.mod['rna']
rna.var['mt'] = rna.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
mu.pp.filter_var(rna, 'n_cells_by_counts', lambda x: x >= 3)
mu.pp.filter_obs(rna, 'n_genes_by_counts', lambda x: (x >= 200) & (x < 5000))

mu.pp.filter_obs(rna, 'total_counts', lambda x: x < 15000)
mu.pp.filter_obs(rna, 'pct_counts_mt', lambda x: x < 20)
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
sc.pl.highly_variable_genes(rna)

rna_matrix = pd.DataFrame.sparse.from_spmatrix(rna.X)
rna_matrix.index = rna.obs.index
rna_matrix.columns = rna.var['gene_ids'].index

# Step 1: Remove all-zero rows and columns from both dataframes
atac_to_expression_dataframe = atac_to_expression_dataframe.loc[:, (atac_to_expression_dataframe != 0).any(axis=0)]
atac_to_expression_dataframe = atac_to_expression_dataframe.loc[(atac_to_expression_dataframe != 0).any(axis=1), :]


# Convert DataFrame to NumPy array
rna_array = rna_matrix.values

# Remove rows and columns with all zeros
non_zero_rows = np.any(rna_array != 0, axis=1)
non_zero_cols = np.any(rna_array != 0, axis=0)

filtered_array = rna_array[non_zero_rows][:, non_zero_cols]

rna_matrix = pd.DataFrame(filtered_array, columns=rna_matrix.columns[non_zero_cols], index=rna_matrix.index[non_zero_rows])


common_genes = set(atac_to_expression_dataframe.columns).intersection(set(rna_matrix.columns))
common_cells = set(atac_to_expression_dataframe.index).intersection(set(rna_matrix.index))


# Convert sets to lists
common_genes = list(common_genes)
common_cells = list(common_cells)

atac_to_expression_dataframe = atac_to_expression_dataframe.loc[common_cells, common_genes]
rna_matrix = rna_matrix.loc[common_cells, common_genes]

# Convert the pandas DataFrame to an AnnData object
adata_rna = sc.AnnData(X=rna_matrix)

sc.pp.highly_variable_genes(adata_rna, n_top_genes=2000, inplace=True)

highly_variable_genes = adata_rna.var[adata_rna.var['highly_variable']].index.tolist()

filtered_rna_matrix = rna_matrix.loc[:, highly_variable_genes]
filtered_atac_to_expression_dataframe = atac_to_expression_dataframe.loc[:, highly_variable_genes]


from mcoia.classes import MCIAnalysis
from mcoia.functions import *

mcia_instance = MCIAnalysis([filtered_atac_to_expression_dataframe, filtered_rna_matrix], nf=20)
mcia_instance.fit()
mcia_instance.transform()
mcia_instance.results()

tco_result = mcia_instance.Tco
df_ana1 = tco_result[tco_result.index.str.endswith('.Ana1')]
df_ana2 = tco_result[tco_result.index.str.endswith('.Ana2')]


df_ana1.index = df_ana1.index.str.replace('.Ana1', '')
df_ana2.index = df_ana1.index.str.replace('.Ana2', '')


df_ana1 = df_ana1.copy()
df_ana1['cell_type'] = df_ana1.index.map(a.obs['predicted.celltype.l1'])

df_ana2 = df_ana2.copy()
df_ana2['cell_type'] = df_ana2.index.map(a.obs['predicted.celltype.l1'])

df_ana1['dataset'] = 'df_ana1'
df_ana2['dataset'] = 'df_ana2'

combined_df = pd.concat([df_ana1, df_ana2])

import umap
import matplotlib.pyplot as plt
import seaborn as sns


umap_model = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='euclidean')
umap_data = umap_model.fit_transform(combined_df.iloc[:, :-2])  # Exclude 'cell_type' and 'dataset' columns

umap_df = pd.DataFrame(data=umap_data, columns=['UMAP1', 'UMAP2'], index=combined_df.index)
umap_df['cell_type'] = combined_df['cell_type']
umap_df['dataset'] = combined_df['dataset']

# Plotting
plt.figure(figsize=(14, 10))
sns.scatterplot(x='UMAP1', y='UMAP2', hue='dataset', style='cell_type', data=umap_df)
plt.title('UMAP Projection')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

############################################################################################




umap_model = umap.UMAP(n_neighbors=25, n_components=2, min_dist=0.25, metric='euclidean')
umap_data = umap_model.fit_transform(combined_df.iloc[:, :-2])  # Exclude 'cell_type' and 'dataset' columns

umap_df = pd.DataFrame(data=umap_data, columns=['UMAP1', 'UMAP2'], index=combined_df.index)
umap_df['cell_type'] = combined_df['cell_type']
umap_df['dataset'] = combined_df['dataset']
marker_dict = {'df_ana1': 'o', 'df_ana2': '^'}

# Plotting
plt.figure(figsize=(14, 10))
sns.scatterplot(x='UMAP1', y='UMAP2', hue='cell_type', style='dataset', palette='tab10', data=umap_df)
plt.title('UMAP Projection')
plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left', title='Cell Type and Dataset')

plt.tight_layout()
plt.show()










###########################################################################################



fig, ax = plt.subplots()

sns.scatterplot(x='SV1', y='SV2', hue='cell_type', style='dataset', data=combined_df, ax=ax)
handles, labels = ax.get_legend_handles_labels()

# Show legend and plot
ax.legend(handles, labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()





