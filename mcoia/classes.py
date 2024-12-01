import pandas as pd
import numpy as np
from .functions.analysis_prep import dataset_validation_and_analysis_prep
from .functions.reformat import compile_tables
from .functions.mcoia import multiple_coinertia_analysis

class MCoIAnalysis:
    def __init__(self, dataset, nf=2, analysis_type = "nsc", is_data_being_projected = False, weight_option = None):
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
        self.nf = nf
        self.analysis_type = analysis_type
        self.TL = None
        self.TC = None
        self.T4 = None
        self.class_type = []
        self.final_results = None
        self.fit()
        self.transform()
        self.results(is_data_being_projected=is_data_being_projected, weight_option = weight_option)

    def fit(self):
        self.multiple_co_inertia_result = dataset_validation_and_analysis_prep(self.dataset, self.nf, analysis_type=self.analysis_type)

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

    def results(self, is_data_being_projected = False, weight_option = None):
        if is_data_being_projected:
            analysis_results = multiple_coinertia_analysis(datasets=self.ktcoa, weight_option = weight_option, n_dim=self.nf, is_data_being_projected=is_data_being_projected)
            return analysis_results

        if self.ktcoa is not None:
            analysis_results = multiple_coinertia_analysis(datasets=self.ktcoa, n_dim=self.nf)
            self.pseudo_eigenvalues = analysis_results['pseudo_eigenvalues']
            self.lambda_df = analysis_results['lambda']
            self.SynVar = analysis_results['SynVar']
            self.axis = analysis_results['axis']
            self.row_projection = analysis_results['row_projection']
            self.cov2 = analysis_results['cov2']
            self.row_projection_normed = analysis_results['row_projection_normed']
            self.column_projection = analysis_results['column_projection']
            self.TL = analysis_results['TL']
            self.TC = analysis_results['TC']
            self.T4 = analysis_results['T4']
            self.class_type = analysis_results['class']

            self.final_results = analysis_results


            return True
        else:
            print("Please transform the model before getting results.")
            return False

    def project(self, projected_dataset):
        analysis_instance = MCoIAnalysis([projected_dataset], self.nf, self.analysis_type, is_data_being_projected=True)
        analysis_results = analysis_instance.results(is_data_being_projected=True)

        weighted_urk = np.array(self.SynVar) * np.array(self.row_weight).T

        projected_coordinates = analysis_results.T.dot(weighted_urk)

        return projected_coordinates

