import pandas as pd
import numpy as np
from .functions.mcia import mcia
from .functions.data_reformat import compile_tables
from .functions.mcoia import multiple_coinertia_analysis

class MCIAnalysis:
    def __init__(self, dataset, nf=2):
        self.dataset = dataset
        self.nf = nf
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
        self.row_projection = pd.DataFrame()
        self.cov2 = pd.DataFrame()
        self.row_projection_normed = pd.DataFrame()
        self.column_projection = pd.DataFrame()
        self.Tax = pd.DataFrame()
        self.nf = nf  # Already present
        self.TL = None
        self.TC = None
        self.T4 = None
        self.class_type = []
        self.final_results = None

    def fit(self):
        self.multiple_co_inertia_result = mcia(self.dataset, self.nf)  # Calling the original mcia function

        if self.multiple_co_inertia_result is not None:
            for res_dict in self.multiple_co_inertia_result:
                self.weighted_table.append(res_dict.get('weighted_table'))
                self.column_weight.append(res_dict.get('column_weight'))
                self.row_weight.append(res_dict.get('row_weight'))
                self.eigenvalues.append(res_dict.get('eigenvalues'))
                self.rank.append(res_dict.get('rank'))
                self.factor_numbers.append(res_dict.get('factor_numbers'))
                self.component_scores.append(res_dict.get('column_scores'))
                self.row_coordinates.append(res_dict.get('row_principal_coordinates'))
                self.principal_coordinates.append(res_dict.get('column_principal_coordinates'))
                self.factor_scores.append(res_dict.get('row_scores'))
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

    def results(self, projected_dataset = False):

        if projected_dataset:
            analysis_results = multiple_coinertia_analysis(X=self.ktcoa, nf=self.nf, data_projected=projected_dataset)
            return analysis_results

        if self.ktcoa is not None:
            analysis_results = multiple_coinertia_analysis(X=self.ktcoa, nf=self.nf)
            self.pseudo_eigenvalues = analysis_results['pseudo_eigenvalues']
            self.lambda_df = analysis_results['lambda']
            self.SynVar = analysis_results['SynVar']
            self.axis = analysis_results['axis']
            self.row_projection = analysis_results['row_projection']
            self.cov2 = analysis_results['cov2']
            self.row_projection_normed = analysis_results['row_projection_normed']
            self.column_projection = analysis_results['column_projection']
            self.Tax = analysis_results['Tax']
            self.TL = analysis_results['TL']
            self.TC = analysis_results['TC']
            self.T4 = analysis_results['T4']
            self.class_type = analysis_results['class']

            # Maybe add these to a final_results dict for easy retrieval
            self.final_results = analysis_results

            # Scale the results (commented out in your original code, uncomment if needed)
            # tab, attributes = scalewt(multiple_co_inertia['Tco'], self.ktcoa['column_weight'], center=False, scale=True)
            # col_names = [f'Axis{i + 1}' for i in range(tab.shape[1])]
            # tab.columns = col_names
            # Assign results to a class attribute

            return True  # Indicate success
        else:
            print("Please transform the model before getting results.")
            return False  # Indicate failure

    def project(self, projected_dataset):
        projected_dataset = MCIAnalysis([projected_dataset])
        projected_dataset.fit()
        projected_dataset.transform()
        tab = projected_dataset.results(projected_dataset = True)
        weighted_urk = np.array(self.SynVar) * np.array(self.row_weight).T
        projected_coordinates = tab.T.dot(weighted_urk)
        return projected_coordinates

