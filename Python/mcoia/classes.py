import pandas as pd
import numpy as np
from numpy.linalg import svd
import itertools
import scipy
from scipy.linalg import eigh
import time
from sklearn.decomposition import TruncatedSVD
from mcoia import mcia, compile_tables, mcoa

class MCIAnalysis:
    def __init__(self, dataset, nf=2, nsc=True):
        self.dataset = dataset
        self.nf = nf
        self.nsc = nsc
        self.multiple_co_inertia_result = None

        # Initialize your attributes as lists
        self.weighted_table = []
        self.column_weight = []
        self.row_weight = []
        self.eigenvalues = []
        self.rank = []
        self.row_names = []
        self.factor_numbers = []
        self.component_scores = []
        self.row_coordinates = []
        self.principal_coordinates = []
        self.factor_scores = []
        self.class_type = []
        self.ktcoa = None
        # self.RV = None
        self.pseudo_eigenvalues = None
        self.lambda_df = pd.DataFrame()
        self.SynVar = pd.DataFrame()
        self.axis = pd.DataFrame()
        self.Tli = pd.DataFrame()
        self.cov2 = pd.DataFrame()
        self.Tl1 = pd.DataFrame()
        self.Tco = pd.DataFrame()
        self.Tax = pd.DataFrame()
        self.nf = nf  # Already present
        self.TL = None
        self.TC = None
        self.T4 = None
        self.class_type = []
        self.final_results = None

    def fit(self):
        self.multiple_co_inertia_result = mcia(self.dataset, self.nf, self.nsc)  # Calling the original mcia function

        if self.multiple_co_inertia_result is not None:
            for res_dict in self.multiple_co_inertia_result:
                self.weighted_table.append(res_dict.get('weighted_table'))
                self.column_weight.append(res_dict.get('column_weight'))
                self.row_weight.append(res_dict.get('row_weight'))
                self.eigenvalues.append(res_dict.get('eigenvalues'))
                self.rank.append(res_dict.get('rank'))
                self.factor_numbers.append(res_dict.get('factor_numbers'))
                self.component_scores.append(res_dict.get('component_scores'))
                self.row_coordinates.append(res_dict.get('row_coordinates'))
                self.principal_coordinates.append(res_dict.get('principal_coordinates'))
                self.factor_scores.append(res_dict.get('factor_scores'))
                self.class_type.append(res_dict.get('dudi'))

            return True
        else:
            return False

    def transform(self):
        if self.multiple_co_inertia_result is not None:
            ktcoa = compile_tables(self.multiple_co_inertia_result)
            self.ktcoa = ktcoa

            return True
        else:
            print("Please fit the model before transforming.")
            return False

    def results(self):

        if self.ktcoa is not None:
            acom = mcoa(X=self.ktcoa, nf=self.nf)
            self.pseudo_eigenvalues = acom['pseudo_eigenvalues']
            self.lambda_df = acom['lambda']
            self.SynVar = acom['SynVar']
            self.axis = acom['axis']
            self.Tli = acom['Tli']
            self.cov2 = acom['cov2']
            self.Tl1 = acom['Tl1']
            self.Tco = acom['Tco']
            self.Tax = acom['Tax']
            self.TL = acom['TL']
            self.TC = acom['TC']
            self.T4 = acom['T4']
            self.class_type = acom['class']

            # Maybe add these to a final_results dict for easy retrieval
            self.final_results = acom

            # Scale the results (commented out in your original code, uncomment if needed)
            # tab, attributes = scalewt(multiple_co_inertia['Tco'], self.ktcoa['column_weight'], center=False, scale=True)
            # col_names = [f'Axis{i + 1}' for i in range(tab.shape[1])]
            # tab.columns = col_names
            # Assign results to a class attribute

            return True  # Indicate success
        else:
            print("Please transform the model before getting results.")
            return False  # Indicate failure