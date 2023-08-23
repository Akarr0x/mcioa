import sys
import numpy as np
import pandas as pd

sys.path.append('/Users/alessandrodiamanti/Desktop/Tesi max planck/Python/mcoia')
from mcoia.mcoia import mcia
np.random.seed(0)

def test_mcia():
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
            -0.705443, 0.113451, 0.579510, -0.519708, -0.445454,
            0.895318, -0.478965, 0.467977, -0.306817, 0.400130,
            -0.594071, -0.015837, 0.917152, 0.047919, -0.369404,
            1.112348, -0.756540, -0.041318, 0.145821, -0.446069
        ],
        'Axis2': [
            -0.284281, -0.632779, 0.241871, -0.667789, 0.124826,
            -0.070683, 0.234842, -0.621184, 0.613006, 1.062170,
            -0.034635, -0.597453, 0.123185, 0.476135, -0.261224,
            0.338772, 0.325667, -1.004717, -0.023100, 0.657370
        ]
    }

    index_values = [
        "Gene_1.df1", "Gene_2.df1", "Gene_3.df1", "Gene_4.df1", "Gene_5.df1",
        "Gene_6.df1", "Gene_7.df1", "Gene_8.df1", "Gene_9.df1", "Gene_10.df1",
        "Gene_1.df2", "Gene_2.df2", "Gene_3.df2", "Gene_4.df2", "Gene_5.df2",
        "Gene_6.df2", "Gene_7.df2", "Gene_8.df2", "Gene_9.df2", "Gene_10.df2"
    ]

    expected_result = pd.DataFrame(expected_result_data, index=index_values)

    pd.testing.assert_frame_equal(result['mcoa']['Tli'], expected_result, atol=1e-6)

def test_mcia_eigenvalues():
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
    expected_result = np.array([
        5.31287029e-01, 4.10235929e-01, 3.87584954e-01, 2.79888554e-01,
        1.67264224e-01, 1.25324059e-01, 6.02087111e-02, 2.09101551e-02,
        3.86080790e-03, 7.15431582e-32
    ])

    np.testing.assert_allclose(result['mcoa']['pseudo_eigenvalues'], expected_result, atol=1e-6)




def test_mcia_tl1():
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
    # Define the expected result
    expected_result_data = {
        'Axis1': [
            -1.328633, 0.213675, 1.091450, -0.978818, -0.838968,
            1.686244, -0.902083, 0.881389, -0.577860, 0.753605,
            -1.024152, -0.027302, 1.581129, 0.082611, -0.636836,
            1.917637, -1.304241, -0.071231, 0.251388, -0.769003
        ],
        'Axis2': [
            -0.523293, -1.164794, 0.445226, -1.229240, 0.229775,
            -0.130110, 0.432288, -1.143450, 1.128398, 1.955200,
            -0.071734, -1.237407, 0.255133, 0.986140, -0.541030,
            0.701643, 0.674500, -2.080905, -0.047843, 1.361503
        ]
    }

    index_values = [
        "Gene_1.df1", "Gene_2.df1", "Gene_3.df1", "Gene_4.df1", "Gene_5.df1",
        "Gene_6.df1", "Gene_7.df1", "Gene_8.df1", "Gene_9.df1", "Gene_10.df1",
        "Gene_1.df2", "Gene_2.df2", "Gene_3.df2", "Gene_4.df2", "Gene_5.df2",
        "Gene_6.df2", "Gene_7.df2", "Gene_8.df2", "Gene_9.df2", "Gene_10.df2"
    ]

    expected_result = pd.DataFrame(expected_result_data, index=index_values)

    pd.testing.assert_frame_equal(result['mcoa']['Tl1'], expected_result, atol=1e-6)


import numpy as np

def test_mcia_random_datasets():
    import time
    start = time.time()

    # Building dataset1 with 1000 observations and positive numbers
    dataset1_values = np.random.randint(1, 2000, size=(2000, 2000)) # Adjust the range as needed
    gene_names = [f"Gene_{i}" for i in range(1, 2001)]
    dataset1 = pd.DataFrame(dataset1_values, columns=gene_names)

    # Building dataset2 with 1000 observations and positive numbers
    dataset2_values = np.random.randint(1, 2000, size=(2000, 2000)) # Adjust the range as needed
    dataset2 = pd.DataFrame(dataset2_values, columns=gene_names)

    data_list = [dataset1, dataset2]

    # Call the mcia function (you should define this function elsewhere in your code)
    result = mcia(data_list)
    print(time.time() - start)
