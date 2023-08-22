import sys
import numpy as np
import pandas as pd
import unittest

sys.path.append('/Users/alessandrodiamanti/Desktop/Tesi max planck/Python/mcioa')
from mcioa.mcioa import mcia
np.random.seed(0)

class TestMCIAFunction(unittest.TestCase):

    def test_mcia(self):
        dataset1_values = [
            [23, 43, 65, 22, 1, 78, 34, 54, 23, 65],
            [34, 23, 45, 65, 23, 43, 56, 67, 34, 23],
            [45, 67, 23, 54, 23, 65, 12, 34, 54, 34],
            [56, 43, 23, 43, 23, 54, 43, 23, 54, 54],
            [67, 65, 34, 65, 12, 43, 34, 65, 23, 12],
            [34, 23, 65, 34, 23, 54, 23, 54, 65, 65],
            [43, 56, 23, 43, 34, 65, 43, 23, 54, 23],
            [54, 45, 67, 65, 34, 54, 23, 65, 23, 34],
            [23, 65, 34, 23, 23, 65, 54, 43, 23, 43],
            [67, 34, 56, 54, 43, 23, 65, 34, 56, 65],
        ]

        dataset2_values = [
            [34, 56, 23, 12, 43, 23, 34, 65, 67, 34],
            [45, 34, 56, 54, 23, 54, 23, 54, 23, 23],
            [65, 43, 23, 43, 34, 23, 54, 43, 34, 65],
            [23, 12, 34, 65, 43, 65, 43, 23, 45, 56],
            [43, 23, 65, 34, 23, 54, 34, 23, 54, 45],
            [12, 34, 23, 43, 54, 65, 23, 54, 65, 23],
            [23, 54, 65, 23, 23, 54, 23, 43, 54, 23],
            [34, 23, 43, 56, 34, 23, 65, 34, 67, 23],
            [45, 65, 23, 45, 23, 54, 43, 23, 45, 67],
            [56, 43, 23, 34, 65, 23, 54, 56, 43, 45],
        ]
        gene_names = [f"Gene_{i}" for i in range(1, 11)]

        # Create DataFrames
        dataset1 = pd.DataFrame(dataset1_values, columns=gene_names, index=gene_names)
        dataset2 = pd.DataFrame(dataset2_values, columns=gene_names, index=gene_names)

        data_list = [dataset1, dataset2]

        # Call the mcia function
        result = mcia(data_list)

        expected_result_data = {
            'Axis1': [
                0.669275, -0.105402, -0.548569, 0.501234, 0.421157,
                -0.845159, 0.444888, -0.435511, 0.288583, -0.390496,
                0.545035, 0.006785, -0.844168, -0.037892, 0.343849,
                -1.027082, 0.701380, 0.039198, -0.131577, 0.404473
            ],
            'Axis2': [
                -0.261830, -0.565188, 0.191639, -0.634826, 0.125566,
                -0.052714, 0.219973, -0.616341, 0.599519, 0.994201,
                -0.053174, -0.522395, 0.087970, 0.402311, -0.226948,
                0.310472, 0.293867, -0.921927, 0.016326, 0.613498
            ]
        }

        index_values = [
            "Gene_1.df1", "Gene_2.df1", "Gene_3.df1", "Gene_4.df1", "Gene_5.df1",
            "Gene_6.df1", "Gene_7.df1", "Gene_8.df1", "Gene_9.df1", "Gene_10.df1",
            "Gene_1.df2", "Gene_2.df2", "Gene_3.df2", "Gene_4.df2", "Gene_5.df2",
            "Gene_6.df2", "Gene_7.df2", "Gene_8.df2", "Gene_9.df2", "Gene_10.df2"
        ]

        expected_result = pd.DataFrame(expected_result_data, index=index_values)

        # Example: assert the result meets some condition
        pd.testing.assert_frame_equal(result['mcoa']['Tli'], expected_result, atol=1e-6)
        # Add other assertions here, depending on what you expect from the function

if __name__ == "__main__":
    unittest.main()



