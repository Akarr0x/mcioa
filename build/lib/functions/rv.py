import pandas as pd
import numpy as np
import itertools


def rv(m1, m2):
    # Convert the datasets to numpy arrays for easier manipulation
    m1, m2 = np.array(m1), np.array(m2)
    # normed_scm1 and normed_scm2 are the "normed sums of cross products" of m1 and m2, respectively.
    normed_scm1 = m1.T @ m1
    normed_scm2 = m2.T @ m2
    # Calculate the RV coefficient using the formula.
    rv_index = np.sum(normed_scm1 * normed_scm2) / np.sqrt(
        np.sum(normed_scm1 * normed_scm2) * np.sum(normed_scm2 * normed_scm2))
    return rv_index


def pairwise_rv(dataset):
    dataset_names = list(dataset.keys())
    n = len(dataset)  # Number of datasets

    # For each combination, call the rv function with the 'weighted_table' of the corresponding datasets.
    RV = [rv(dataset[dataset_names[i]]['weighted_table'].values, dataset[dataset_names[j]]['weighted_table'].values)
          for i, j in itertools.combinations(range(n), 2)]

    m = np.ones((n, n))
    m[np.tril_indices(n, -1)] = RV
    m[np.triu_indices(n, 1)] = RV

    m = pd.DataFrame(m)
    # m.columns = m.index = list_of_names

    return m