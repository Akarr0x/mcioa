import sys
import random

import matplotlib.pyplot as plt
import numpy as np

# sys.path.append("/Users/alessandrodiamanti/Desktop/Tesi max planck/Python/mcoia/mcoia.py")
sys.path.append("/Users/alessandrodiamanti/Desktop/Tesi max planck/Python/mcoia")

from scipy.io import mmread
import pandas as pd
from classes import MCIAnalysis
import matplotlib.pyplot as plt



first = pd.read_csv("/Users/alessandrodiamanti/Downloads/ifnb_stim_matrix-2.csv", header=0, index_col=0,
                      sep=',')
second = pd.read_csv("/Users/alessandrodiamanti/Downloads/ifnb_ctrl_matrix-2.csv", header= 0, index_col = 0,
                      sep=',')


data_list = [(np.log2(first+1)).T, (np.log(second+1)).T]

mcia_instance = MCIAnalysis(data_list, nf = 10)

mcia_instance.fit()

mcia_instance.transform()

mcia_instance.results()

Tco =mcia_instance.Tco


num_datasets = 2  # replace N with the number of datasets you want
chunk_size = mcia_instance.Tco.shape[0] // num_datasets

# Initialize list of lists for SV1 and SV2 values
x = [[] for _ in range(num_datasets)]
y = [[] for _ in range(num_datasets)]

# Loop through DataFrame and distribute rows to different datasets
for i in range(mcia_instance.Tco.shape[0]):
    dataset_index = i // chunk_size
    if dataset_index >= num_datasets:
        dataset_index = num_datasets - 1  # Put any overflow rows in the last dataset
    x[dataset_index].append(mcia_instance.Tco.iloc[i]['SV1'])
    y[dataset_index].append(mcia_instance.Tco.iloc[i]['SV2'])

