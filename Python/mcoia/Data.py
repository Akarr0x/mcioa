import sys
import random

import matplotlib.pyplot as plt
import numpy as np

# sys.path.append("/Users/alessandrodiamanti/Desktop/Tesi max planck/Python/mcoia/mcoia.py")
sys.path.append("/Users/alessandrodiamanti/Desktop/Tesi max planck/Python/mcoia")

from scipy.io import mmread
import pandas as pd
# from mcoia import mcia
from classes import MCIAnalysis


# Step 1: Remove rows with all zeros in each DataFrame
def remove_zero_rows(df):
    return df.loc[(df != 0).any(axis=1)]


# Step 2: Remove columns with all zeros in each DataFrame
def remove_zero_columns(df):
    return df.loc[:, (df != 0).any(axis=0)]


# Step 3: Remove duplicate rows by taking the mean
def remove_duplicate_rows(df):
    return df.groupby(df.index).mean()


# Load the data
matrix = mmread('/Users/alessandrodiamanti/Downloads/filtered_feature_bc_matrix/matrix.mtx')
dense_matrix = matrix.toarray()

# Load the genes and cells
genes = pd.read_csv('/Users/alessandrodiamanti/Downloads/filtered_feature_bc_matrix/features.tsv', header=None,
                    sep='\t')

df = pd.DataFrame(dense_matrix, index=genes[1])
df = remove_zero_rows(df)
df = remove_zero_columns(df)
df = remove_duplicate_rows(df)
# Remove rows where the sum along the row is zero
df = df[df.sum(axis=1) != 0]
# Remove columns where the sum along the column is zero
df = df.loc[:, df.sum(axis=0) != 0]

# Load the second matrix
matrix_2 = mmread('/Users/alessandrodiamanti/Downloads/filtered_feature_bc_matrix 7/matrix.mtx')
dense_matrix_2 = matrix_2.toarray()

# Load the second genes and cells
genes_2 = pd.read_csv('/Users/alessandrodiamanti/Downloads/filtered_feature_bc_matrix 7/features.tsv', header=None,
                      sep='\t')

df_2 = pd.DataFrame(dense_matrix_2, index=genes_2[1])
df_2 = remove_zero_rows(df_2)
df_2 = remove_zero_columns(df_2)
df_2 = remove_duplicate_rows(df_2)

df = df.groupby(df.index).mean()
df_2 = df_2.groupby(df_2.index).mean()

df = df.loc[(df.sum(axis=1) != 0)]
df_2 = df_2.loc[(df_2.sum(axis=1) != 0)]

# Remove columns with all zeros
df = df.loc[:, (df.sum(axis=0) != 0)]
df_2 = df_2.loc[:, (df_2.sum(axis=0) != 0)]

# Eliminate duplicates by averaging them

# Find common genes and filter both DataFrames
common_genes = df.index.intersection(df_2.index)
df_1_filtered = df.loc[common_genes]
df_2_filtered = df_2.loc[common_genes]

# Remove any newly zero rows after the intersection
df_1_filtered = df_1_filtered.loc[(df_1_filtered.sum(axis=1) != 0)]
df_2_filtered = df_2_filtered.loc[(df_2_filtered.sum(axis=1) != 0)]

# Find the DataFrame with the least number of columns (individuals)
min_columns = min(df_1_filtered.shape[1], df_2_filtered.shape[1])

# Trim both DataFrames to have the same number of columns
df_1_filtered = df_1_filtered.iloc[:, :min_columns]
df_2_filtered = df_2_filtered.iloc[:, :min_columns]

# Debugging: Check if any row has all zero values
if any(df_1_filtered.sum(axis=1) == 0):
    print("Zero-expression rows found in the first dataset.")

if any(df_2_filtered.sum(axis=1) == 0):
    print("Zero-expression rows found in the second dataset.")

# Calculate the variance for each gene in each DataFrame
variances_1 = df_1_filtered.var(axis=1)
variances_2 = df_2_filtered.var(axis=1)

# Sort the variances and take the top 25% most variable genes for each DataFrame
top_25_percent_indices_1 = variances_1.sort_values(ascending=False).index[:len(variances_1)//4]
top_25_percent_indices_2 = variances_2.sort_values(ascending=False).index[:len(variances_2)//4]

# Filter the DataFrames to only include the top 25% most variable genes
df_1_filtered_top = df_1_filtered.loc[top_25_percent_indices_1]
df_2_filtered_top = df_2_filtered.loc[top_25_percent_indices_2]

# Intersect the genes between the two DataFrames to have a common set for comparison
common_genes = df_1_filtered_top.index.intersection(df_2_filtered_top.index)

df_1_filtered_common = df_1_filtered_top.loc[common_genes]
df_2_filtered_common = df_2_filtered_top.loc[common_genes]




data_list = [df_1_filtered_common, df_2_filtered_common]

mcia_instance = MCIAnalysis(data_list)

mcia_instance.fit()

mcia_instance.transform()

mcia_instance.results()

tco_result = mcia_instance.Tco

num_datasets = 2  # replace N with the number of datasets you want
chunk_size = tco_result.shape[0] // num_datasets

# Initialize list of lists for SV1 and SV2 values
x = [[] for _ in range(num_datasets)]
y = [[] for _ in range(num_datasets)]

# Loop through DataFrame and distribute rows to different datasets
for i in range(tco_result.shape[0]):
    dataset_index = i // chunk_size
    if dataset_index >= num_datasets:
        dataset_index = num_datasets - 1  # Put any overflow rows in the last dataset
    x[dataset_index].append(tco_result.iloc[i]['SV1'])
    y[dataset_index].append(tco_result.iloc[i]['SV2'])

import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x[0], y[0], s=10, c="b", marker="s", label="first")
ax1.scatter(x[1], y[1], s=10, c="r", marker="o", label="second")
plt.show()



first = pd.read_csv("/Users/alessandrodiamanti/Downloads/ifnb_stim_matrix-2.csv", header=0, index_col=0,
                      sep=',')
second = pd.read_csv("/Users/alessandrodiamanti/Downloads/ifnb_ctrl_matrix-2.csv", header= 0, index_col = 0,
                      sep=',')


data_list = [(np.log2(first+1)).T, (np.log(second+1)).T]

mcia_instance = MCIAnalysis(data_list, nf = 10)

mcia_instance.fit()

mcia_instance.transform()

mcia_instance.results()

mcia_instance.Tco

