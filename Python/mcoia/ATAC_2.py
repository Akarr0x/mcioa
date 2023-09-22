########################################################################################################################


import anndata
import numpy as np
import pandas as pd
import scanpy as sc
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


atac_to_expression_dataframe = atac_to_expression_dataframe.loc[:, (atac_to_expression_dataframe != 0).any(axis=0)]
atac_to_expression_dataframe = atac_to_expression_dataframe.loc[(atac_to_expression_dataframe != 0).any(axis=1), :]

rna_array = rna_matrix.values

non_zero_rows = np.any(rna_array != 0, axis=1)
non_zero_cols = np.any(rna_array != 0, axis=0)

filtered_array = rna_array[non_zero_rows][:, non_zero_cols]

rna_matrix = pd.DataFrame(filtered_array, columns=rna_matrix.columns[non_zero_cols], index=rna_matrix.index[non_zero_rows])


common_genes = set(atac_to_expression_dataframe.columns).intersection(set(rna_matrix.columns))
common_cells = set(atac_to_expression_dataframe.index).intersection(set(rna_matrix.index))


common_genes = list(common_genes)
common_cells = list(common_cells)

atac_to_expression_dataframe = atac_to_expression_dataframe.loc[common_cells, common_genes]
rna_matrix = rna_matrix.loc[common_cells, common_genes]

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

ax.legend(handles, labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()





##########################################################################################
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx


df_ana1_SV = df_ana1.iloc[:, :-2]
df_ana2_SV = df_ana2.iloc[:, :-2]

# Find k-NN from df_ana1 to df_ana2
k = 25
nbrs_1_to_2 = NearestNeighbors(n_neighbors=k).fit(df_ana2_SV)
distances_1_to_2, indices_1_to_2 = nbrs_1_to_2.kneighbors(df_ana1_SV)

# Find k-NN from df_ana2 to df_ana1
nbrs_2_to_1 = NearestNeighbors(n_neighbors=k).fit(df_ana1_SV)
distances_2_to_1, indices_2_to_1 = nbrs_2_to_1.kneighbors(df_ana2_SV)

G = nx.Graph()
for i, cell_1 in enumerate(df_ana1.index):
    for j in indices_1_to_2[i]:
        cell_2 = df_ana2.index[j]
        if i in indices_2_to_1[j]:
            G.add_edge(cell_1 + '_ana1', cell_2 + '_ana2')

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=20, font_size=8)
plt.show()




# Extract only the columns that have feature vectors (ignoring 'cell_type' and 'dataset' for now)
df_ana1_SV = df_ana1.iloc[:, :-2]
df_ana2_SV = df_ana2.iloc[:, :-2]

##############################################################################

# Number of nearest neighbors to find for each point
k = 30

# Initialize the NearestNeighbors algorithm
nbrs = NearestNeighbors(n_neighbors=k).fit(df_ana2_SV)

# Find the k nearest neighbors from df_ana2 for each point in df_ana1
distances, indices = nbrs.kneighbors(df_ana1_SV)

# Initialize a bipartite graph
G = nx.Graph()

# Add edges to the bipartite graph
for i, cell_1 in enumerate(df_ana1_SV.index):
    for j in indices[i]:
        cell_2 = df_ana2_SV.index[j]
        G.add_edge(cell_1 + '_ana1', cell_2 + '_ana2')

# Draw the bipartite graph
# plt.figure(figsize=(12, 12))
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=15, font_size=8)
# plt.show()


partition = nx.community.louvain_communities(G)

partition_dict = {}

# Partition dictionary will have the communities for each cell
for community, nodes in enumerate(partition):
    for node in nodes:
        partition_dict[node] = community

# Differentiate the nodes colors based on the partition
partition_values = [partition_dict.get(node) for node in G.nodes()]

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=partition_values, node_size=10, alpha=0.8)





##############################################################################
import networkx as nx

G = nx.Graph()

# Assuming combined_df_SV contains only the feature columns
combined_df_SV = combined_df.iloc[:, :-2]

# Find k-NN for combined_df_SV
k = 25
nbrs = NearestNeighbors(n_neighbors=k).fit(combined_df_SV)
distances, indices = nbrs.kneighbors(combined_df_SV)

# Create k-NN Graph
G = nx.Graph()
for i, cell_1 in enumerate(combined_df_SV.index):
    for j in indices[i]:
        cell_2 = combined_df_SV.index[j]
        G.add_edge(cell_1, cell_2)

# Plot Graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=15, font_size=8)
plt.show()




# Create a new dictionary that only contains the keys that end with '_ana1'
partition_dict_ana1 = {k.replace('_ana1', ''): v for k, v in partition_dict.items() if '_ana1' in k}


df_ana1_SV['louvain_cluster'] = df_ana1_SV.index.map(partition_dict_ana1)

# UMAP on df_ana1_SV
umap_model_SV = umap.UMAP()
umap_data_SV = umap_model_SV.fit_transform(df_ana1_SV.drop(columns=['louvain_cluster']))
umap_df_SV = pd.DataFrame(umap_data_SV, columns=['UMAP1_SV', 'UMAP2_SV'], index=df_ana1_SV.index)
umap_df_SV['louvain_cluster'] = df_ana1_SV['louvain_cluster']

# UMAP on df_ana1
umap_model = umap.UMAP()
umap_data = umap_model.fit_transform(df_ana1.iloc[:, :-3])  # Adjust the slice as needed
umap_df = pd.DataFrame(umap_data, columns=['UMAP1', 'UMAP2'], index=df_ana1.index)
umap_df['cell_type'] = df_ana1['cell_type']


fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# Plot UMAP of df_ana1_SV
sns.scatterplot(x='UMAP1_SV', y='UMAP2_SV', hue='louvain_cluster', palette='tab10', data=umap_df_SV, ax=axs[0])
axs[0].set_title('UMAP of df_ana1_SV colored by Louvain cluster')

# Plot UMAP of df_ana1
sns.scatterplot(x='UMAP1', y='UMAP2', hue='cell_type', palette='tab10', data=umap_df, ax=axs[1])
axs[1].set_title('UMAP of df_ana1 colored by cell type')

plt.tight_layout()
plt.show()



#############################################################################################

combined_df_SV = pd.concat([df_ana1_SV, df_ana2_SV])
k = 30
nbrs = NearestNeighbors(n_neighbors=k).fit(combined_df_SV)
distances, indices = nbrs.kneighbors(combined_df_SV)

partition_list = list(nx.community.louvain_communities(G))


partition_dict = {}
for community, nodes in enumerate(partition_list):
    for node in nodes:
        partition_dict[node] = community


# Update DataFrames with Louvain cluster labels
df_ana1_SV['louvain_cluster'] = df_ana1_SV.index.map(partition_dict.get)
df_ana2_SV['louvain_cluster'] = df_ana2_SV.index.map(partition_dict.get)


plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=list(partition.values()), node_size=15, cmap=plt.cm.jet)
plt.show()





import pandas as pd
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt
import leidenalg
import igraph as ig

# Concatenate your dataframes
combined_df_SV = pd.concat([df_ana1_SV, df_ana2_SV])

# K-nearest neighbors
k = 30
nbrs = NearestNeighbors(n_neighbors=k).fit(combined_df_SV)
distances, indices = nbrs.kneighbors(combined_df_SV)

# Build the NetworkX graph
G = nx.Graph()
for i, index_1 in enumerate(combined_df_SV.index):
    for j in indices[i][1:]:
        index_2 = combined_df_SV.index[j]
        G.add_edge(index_1, index_2)

# Convert the NetworkX graph to an iGraph object
G_ig = ig.Graph.Adjacency((nx.to_numpy_array(G) > 0).tolist())
G_ig.to_undirected()

# Run Leiden algorithm
partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)

# Map node to community
partition_dict = {}
for idx, node in enumerate(G.nodes()):
    partition_dict[node] = partition.membership[idx]

# Update your dataframe
df_ana1_SV['leiden_cluster'] = df_ana1_SV.index.map(partition_dict.get)
df_ana2_SV['leiden_cluster'] = df_ana2_SV.index.map(partition_dict.get)


print(df_ana1_SV['leiden_cluster'].isna().sum(), df_ana2_SV['leiden_cluster'].isna().sum())

# Plotting
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=[partition_dict[node] for node in G.nodes()], node_size=15, cmap=plt.cm.jet)
plt.show()






#####################################################################

# Concatenating the modified dataframes with 'louvain_cluster'
combined_df_SV = pd.concat([df_ana1_SV, df_ana2_SV])
combined_df_SV = combined_df_SV.drop('louvain_cluster', axis = 1)

# Running UMAP on the concatenated dataset
umap_model = umap.UMAP()
umap_data = umap_model.fit_transform(combined_df_SV.drop(columns=['leiden_cluster']))
umap_df = pd.DataFrame(umap_data, columns=['UMAP1', 'UMAP2'], index=combined_df_SV.index)

# Adding the 'louvain_cluster' column to umap_df
umap_df['leiden_cluster'] = combined_df_SV['leiden_cluster']

# Assuming df_ana1 and df_ana2 have a 'cell_type' column
cell_type_dict_ana1 = df_ana1['cell_type'].to_dict()
cell_type_dict_ana2 = df_ana2['cell_type'].to_dict()

# Merge the two dictionaries
cell_type_dict = {**cell_type_dict_ana1, **cell_type_dict_ana2}

# Adding the 'cell_type' column to umap_df
umap_df['cell_type'] = umap_df.index.map(cell_type_dict)

# Create side-by-side UMAP plots
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Plot UMAP colored by Louvain cluster
sns.scatterplot(x='UMAP1', y='UMAP2', hue='leiden_cluster', palette='deep', data=umap_df, ax=axs[0])
axs[0].set_title('UMAP colored by Louvain cluster')

# Plot UMAP colored by cell type
sns.scatterplot(x='UMAP1', y='UMAP2', hue='cell_type', palette='deep', data=umap_df, ax=axs[1])
axs[1].set_title('UMAP colored by cell type')

plt.tight_layout()
plt.show()




###################################################################


combined_df_SV = pd.concat([df_ana1_SV, df_ana2_SV])

k = 30
nbrs = NearestNeighbors(n_neighbors=k).fit(combined_df_SV)
distances, indices = nbrs.kneighbors(combined_df_SV)

# Build the graph
G = nx.Graph()
for i, index_1 in enumerate(combined_df_SV.index):
    for j in indices[i][1:]:  # Exclude the first neighbor, which is the point itself
        index_2 = combined_df_SV.index[j]
        G.add_edge(index_1, index_2)

# Apply the Louvain community detection algorithm
partition_list = list(nx.community.louvain_communities(G))

# Convert the list of sets into a dictionary that maps each node to its community
partition_dict = {}
for community, nodes in enumerate(partition_list):
    for node in nodes:
        partition_dict[node] = community

# Update DataFrames with Louvain cluster labels
df_ana1_SV['louvain_cluster'] = df_ana1_SV.index.map(partition_dict.get)
df_ana2_SV['louvain_cluster'] = df_ana2_SV.index.map(partition_dict.get)

# Plotting
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)
#nx.draw(G, pos, node_color=list(partition_dict.values()), node_size=15, cmap=plt.cm.jet)
#plt.show()

import matplotlib.cm as cm

# Get the number of unique clusters and cell types
n_clusters = umap_df['louvain_cluster'].nunique()
n_cell_types = umap_df['cell_type'].nunique()

# Generate a color map
colors_cluster = [cm.viridis(x) for x in np.linspace(0, 1, n_clusters)]
colors_cell_type = [cm.viridis(x) for x in np.linspace(0, 1, n_cell_types)]


fig, axs = plt.subplots(1, 2, figsize=(14, 7))

sns.scatterplot(x='UMAP1', y='UMAP2', hue='louvain_cluster', palette=colors_cluster, data=umap_df, ax=axs[0])
axs[0].set_title('UMAP colored by Louvain cluster')

sns.scatterplot(x='UMAP1', y='UMAP2', hue='cell_type', palette=colors_cell_type, data=umap_df, ax=axs[1])
axs[1].set_title('UMAP colored by cell type')

plt.tight_layout()
plt.show()





###########################################################


# Initialize an empty NetworkX graph
G = nx.Graph()

# Run K-NN for df_ana1_SV against df_ana2_SV
nbrs_ana1 = NearestNeighbors(n_neighbors=k).fit(df_ana2_SV)
distances1, indices1 = nbrs_ana1.kneighbors(df_ana1_SV)

# Add edges from df_ana1_SV to its k nearest neighbors in df_ana2_SV
for i, index_1 in enumerate(df_ana1_SV.index):
    for j in indices1[i]:
        index_2 = df_ana2_SV.index[j]
        G.add_edge(index_1, index_2)

# Run K-NN for df_ana2_SV against df_ana1_SV
nbrs_ana2 = NearestNeighbors(n_neighbors=k).fit(df_ana1_SV)
distances2, indices2 = nbrs_ana2.kneighbors(df_ana2_SV)

# Add edges from df_ana2_SV to its k nearest neighbors in df_ana1_SV
for i, index_1 in enumerate(df_ana2_SV.index):
    for j in indices2[i]:
        index_2 = df_ana1_SV.index[j]
        G.add_edge(index_1, index_2)


# Convert the NetworkX graph to iGraph adjacency matrix
G_ig = ig.Graph.Adjacency((nx.to_numpy_array(G) > 0).tolist())
G_ig.to_undirected()


partition_list = list(nx.community.louvain_communities(G))


partition_dict = {}
for community, nodes in enumerate(partition_list):
    for node in nodes:
        partition_dict[node] = community

# Update DataFrames with Louvain cluster labels
df_ana1_SV['louvain_cluster'] = df_ana1_SV.index.map(partition_dict.get)
df_ana2_SV['louvain_cluster'] = df_ana2_SV.index.map(partition_dict.get)

# Plotting
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)
#nx.draw(G, pos, node_color=list(partition_dict.values()), node_size=15, cmap=plt.cm.jet)
#plt.show()

import matplotlib.cm as cm

df_ana1_SV['dataset'] = 'SV1'
df_ana2_SV['dataset'] = 'SV2'
combined_df_SV = pd.concat([df_ana1_SV, df_ana2_SV])

# Running UMAP on the concatenated dataset
umap_model = umap.UMAP()
umap_data = umap_model.fit_transform(combined_df_SV.drop(columns=['louvain_cluster', 'dataset']))
umap_df = pd.DataFrame(umap_data, columns=['UMAP1', 'UMAP2'], index=combined_df_SV.index)

# Adding the 'leiden_cluster' and 'dataset' columns to umap_df
umap_df['louvain_cluster'] = combined_df_SV['louvain_cluster']
umap_df['dataset'] = combined_df_SV['dataset']

fig, axs = plt.subplots(1, 2, figsize=(14, 7))
umap_df['cell_type'] = umap_df.index.map(cell_type_dict)

# Plot UMAP colored by Leiden cluster
sns.scatterplot(x='UMAP1', y='UMAP2', hue='louvain_cluster', palette='deep', data=umap_df, ax=axs[0])
axs[0].set_title('UMAP colored by louvain cluster')

# Plot UMAP colored by both Leiden cluster and Dataset (via markers)
sns.scatterplot(x='UMAP1', y='UMAP2', hue='cell_type', palette='deep', data=umap_df, ax=axs[1])
axs[1].set_title('UMAP colored by cell type')

plt.tight_layout()
plt.show()









#######################################################








k = 50





df_ana1_SV = df_ana1_SV.drop('louvain_cluster', axis = 1)
df_ana2_SV = df_ana2_SV.drop('louvain_cluster', axis = 1)

df_ana1_SV = df_ana1_SV.drop('dataset', axis = 1)
df_ana2_SV = df_ana2_SV.drop('dataset', axis = 1)

G = nx.Graph()

# Run K-NN for df_ana1_SV against df_ana2_SV
nbrs_ana1 = NearestNeighbors(n_neighbors=k).fit(df_ana2_SV)
distances1, indices1 = nbrs_ana1.kneighbors(df_ana1_SV)

# Add edges from df_ana1_SV to its k nearest neighbors in df_ana2_SV
for i, index_1 in enumerate(df_ana1_SV.index):
    for j in indices1[i]:
        index_2 = df_ana2_SV.index[j]
        G.add_edge(index_1, index_2)



# Run K-NN for df_ana2_SV against df_ana1_SV
nbrs_ana2 = NearestNeighbors(n_neighbors=k).fit(df_ana1_SV)
distances2, indices2 = nbrs_ana2.kneighbors(df_ana2_SV)

# Add edges from df_ana2_SV to its k nearest neighbors in df_ana1_SV
for i, index_1 in enumerate(df_ana2_SV.index):
    for j in indices2[i]:
        index_2 = df_ana1_SV.index[j]
        G.add_edge(index_1, index_2)



G_ig = ig.Graph.Adjacency((nx.to_numpy_array(G) > 0).tolist())
G_ig.to_undirected()

# Run Leiden algorithm
partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)


partition_dict = {}
for idx, node in enumerate(G.nodes()):
    partition_dict[node] = partition.membership[idx]


df_ana1_SV['leiden_cluster'] = df_ana1_SV.index.map(partition_dict.get)
df_ana2_SV['leiden_cluster'] = df_ana2_SV.index.map(partition_dict.get)


df_ana1_SV['dataset'] = 'SV1'
df_ana2_SV['dataset'] = 'SV2'
combined_df_SV = pd.concat([df_ana1_SV, df_ana2_SV])


# Running UMAP on the concatenated dataset
umap_model = umap.UMAP(min_dist=0.2, n_neighbors=15)
umap_data = umap_model.fit_transform(combined_df_SV.drop(columns=['leiden_cluster', 'dataset']))
umap_df = pd.DataFrame(umap_data, columns=['UMAP1', 'UMAP2'], index=combined_df_SV.index)


umap_df['leiden_cluster'] = combined_df_SV['leiden_cluster']
umap_df['dataset'] = combined_df_SV['dataset']

fig, axs = plt.subplots(1, 2, figsize=(14, 7))
umap_df['cell_type'] = umap_df.index.map(cell_type_dict)


sns.scatterplot(x='UMAP1', y='UMAP2', hue='leiden_cluster', palette='deep', data=umap_df, ax=axs[0])
axs[0].set_title('UMAP colored by leiden cluster')


sns.scatterplot(x='UMAP1', y='UMAP2', hue='cell_type', palette='deep', data=umap_df, ax=axs[1])
axs[1].set_title('UMAP colored by cell type')

plt.tight_layout()
plt.show()















