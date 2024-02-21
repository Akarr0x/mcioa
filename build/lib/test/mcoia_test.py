import sys
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from mcoia.classes import MCIAnalysis
from mcoia.functions import *


np.random.seed(0)


def test_mcia_tli():
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

    mcia_instance = MCIAnalysis(data_list)

    mcia_instance.fit()

    mcia_instance.transform()

    mcia_instance.results()

    Tli = mcia_instance.row_projection

    # Call the mcia function
    result = mcia(data_list)

    expected_result_data = {
        'Axis1': [
            -0.6692748076562199, 0.1054021228158305, 0.5485693098917369,
            -0.5012336431142312, -0.4211566987506613, 0.8451587791535304,
            -0.4448883365216628, 0.43551082686018505, -0.2885834901540539,
            0.39049593747554623, -0.5450346095014146, -0.006784942985822237,
            0.8441684098519833, 0.03789247612277662, -0.34384901435831383,
            1.027082236724715, -0.7013795386726113, -0.03919846248740025,
            0.13157660648023647, -0.404473161174149
        ],
        'Axis2': [
            -0.2618304588466546, -0.5651877902685526, 0.19163873571485326,
            -0.6348255279110901, 0.12556641686630826, -0.052713729888381744,
            0.21997300965454253, -0.6163405031564844, 0.599519154308491,
            0.9942006935269699, -0.05317391735013032, -0.5223954110060904,
            0.08797015211935609, 0.4023111457851837, -0.22694807624639823,
            0.3104721893635913, 0.2938671768523037, -0.9219273296061783,
            0.01632583452766171, 0.6134982355607008
        ]
    }

    index_values = [
        "Gene_1.df1", "Gene_2.df1", "Gene_3.df1", "Gene_4.df1", "Gene_5.df1",
        "Gene_6.df1", "Gene_7.df1", "Gene_8.df1", "Gene_9.df1", "Gene_10.df1",
        "Gene_1.df2", "Gene_2.df2", "Gene_3.df2", "Gene_4.df2", "Gene_5.df2",
        "Gene_6.df2", "Gene_7.df2", "Gene_8.df2", "Gene_9.df2", "Gene_10.df2"
    ]

    expected_result = pd.DataFrame(expected_result_data, index=index_values)

    pd.testing.assert_frame_equal(Tli, expected_result, atol=1e-6)


def test_mcia_tco_pca():

    dataset1_values = [
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [2, 3, 4, 5, 6],
        [6, 5, 4, 3, 2]
    ]

    dataset2_values = [
        [10, 9, 8, 7, 6],
        [6, 7, 8, 9, 10],
        [7, 8, 9, 10, 11],
        [11, 10, 9, 8, 7]
    ]
    gene_names = [f"Gene_{i}" for i in range(1, 6)]
    cell_names = [f"Cell{i}" for i in range(1, 5)]
    # Create DataFrames
    dataset1 = pd.DataFrame(dataset1_values, columns=gene_names, index=cell_names)
    dataset2 = pd.DataFrame(dataset2_values, columns=gene_names, index=cell_names)

    data_list = [dataset1, dataset2]

    mcia_instance = MCIAnalysis(data_list)

    mcia_instance.fit(analysis_type="pca")

    mcia_instance.transform()

    mcia_instance.results()

    tco_result = mcia_instance.column_projection

    expected_result_data = {
        'SV1': [-0.8345481, 0.8345481, -0.8345481, 0.8345481,
                         0.8345481, -0.8345481, -0.8345481, 0.8345481],

        'SV2': [-0.550935, -0.550935, 0.550935, 0.550935,
                         -0.550935, -0.550935, 0.550935, 0.550935]
    }

    expected_result = pd.DataFrame(expected_result_data).reset_index(drop=True)

    def normalize_sign(df, column):
        sign = np.sign(df[column].loc[df[column].ne(0).idxmax()])
        df[column] *= sign
        return df

    tco_result_normalized = normalize_sign(tco_result.copy(), "SV1")
    expected_result_normalized = normalize_sign(expected_result.copy(), "SV1")

    tco_result_normalized.reset_index(drop=True, inplace=True)
    expected_result_normalized.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(tco_result_normalized, expected_result_normalized, atol=1e-6)



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


    mcia_instance = MCIAnalysis(data_list)

    mcia_instance.fit()

    mcia_instance.transform()

    mcia_instance.results()

    pseudo_eigenvalues_result = mcia_instance.pseudo_eigenvalues

    expected_result = np.array([
        4.619700e-01, 3.591855e-01, 3.404723e-01, 2.420725e-01,
        1.457940e-01, 1.085012e-01, 5.293218e-02, 1.824309e-02,
        3.427896e-03, 1.675307e-32
    ])

    np.testing.assert_allclose(pseudo_eigenvalues_result, expected_result, atol=1e-6)




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

    mcia_instance = MCIAnalysis(data_list)

    mcia_instance.fit()

    mcia_instance.transform()

    mcia_instance.results()

    tl1_result = mcia_instance.row_projection_normed

    # Define the expected result
    expected_result_data = {
        'Axis1': [
            -1.3315821045577074, 0.20970695283669874, 1.0914277181886987, -0.997249174386987,
            -0.8379289297245524, 1.6815190007999725, -0.8851451461519655, 0.866487751748806,
            -0.5741626709447716, 0.776926602191808, -1.0193929753857613, -0.012690062424599352,
            1.5788710148753347, 0.07087132322636645, -0.6431101139629064, 1.9209796937820933,
            -1.3118091261327696, -0.07331394485596311, 0.24609128674185893, -0.7564970958636532
        ],
        'Axis2': [
            -0.5112950827358473, -1.1036826626649079, 0.37422667960149364, -1.2396692586717972,
            0.24520253213971435, -0.10293787439138355, 0.429557044915249, -1.2035722273985348,
            1.1707239751789544, 1.941446874027062, -0.12179172943864196, -1.196515956090002,
            0.20149045809525135, 0.9214700112272064, -0.5198112171578398, 0.7111182844816425,
            0.6730854802068781, -2.111620310953264, 0.037393363527469, 1.4051816161013004
        ]
    }

    index_values = [
        "Gene_1.df1", "Gene_2.df1", "Gene_3.df1", "Gene_4.df1", "Gene_5.df1",
        "Gene_6.df1", "Gene_7.df1", "Gene_8.df1", "Gene_9.df1", "Gene_10.df1",
        "Gene_1.df2", "Gene_2.df2", "Gene_3.df2", "Gene_4.df2", "Gene_5.df2",
        "Gene_6.df2", "Gene_7.df2", "Gene_8.df2", "Gene_9.df2", "Gene_10.df2"
    ]

    expected_result = pd.DataFrame(expected_result_data, index=index_values)

    pd.testing.assert_frame_equal(tl1_result, expected_result, atol=1e-6)



def test_mcia_tco():
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
        [67, 34, 56, 54, 43, 23, 65, 34, 56, 65],
    ]
    gene_names = [f"Gene_{i}" for i in range(1, 12)]
    col_names = [f"Cell_{i}" for i in range(1, 11)]

    # Create DataFrames
    dataset1 = pd.DataFrame(dataset1_values, columns=col_names, index=gene_names)
    dataset2 = pd.DataFrame(dataset2_values, columns=col_names, index=gene_names)

    data_list = [dataset1, dataset2]

    mcia_instance = MCIAnalysis(data_list)

    mcia_instance.fit()

    mcia_instance.transform()

    mcia_instance.results()

    tco_result = mcia_instance.column_projection
    # Define the expected result
    expected_result_data = {
        'SV1': [
            0.79377280, -0.06172944, 0.48460118, -0.28990046, 0.16570383,
            0.17442810, 0.14524457, 0.26837507, 0.43542302, -0.88465755,
            -0.88465755, 0.09480383, 0.57792905, -0.62126609, 0.05318700,
            0.17899982, 0.72320175, 0.88071027, -0.46612965, 0.05507923,
            -0.48569217, -0.67776275
        ],
        'SV2': [
            0.80115759, -0.14597006, -0.44426367, 0.03584057, -0.88514828,
            0.67417803, -0.38227838, -0.09468216, -0.07623691, 0.20717265,
            0.20717265, -0.63091762, -0.02660173, -0.15542932, 0.55284894,
            0.70578323, -0.30482621, 0.12592331, -0.03306458, 0.01904726,
            -0.49472329, 0.22601840
        ]
    }

    index_values = [
        "Gene_1.Ana1", "Gene_2.Ana1", "Gene_3.Ana1", "Gene_4.Ana1", "Gene_5.Ana1",
        "Gene_6.Ana1", "Gene_7.Ana1", "Gene_8.Ana1", "Gene_9.Ana1", "Gene_10.Ana1", "Gene_11.Ana1",
        "Gene_1.Ana2", "Gene_2.Ana2", "Gene_3.Ana2", "Gene_4.Ana2", "Gene_5.Ana2",
        "Gene_6.Ana2", "Gene_7.Ana2", "Gene_8.Ana2", "Gene_9.Ana2", "Gene_10.Ana2", "Gene_11.Ana2",
    ]

    expected_result = pd.DataFrame(expected_result_data, index=index_values)

    assert np.allclose(tco_result, expected_result, atol=1e-6)


def test_mcia_tco_2():
    dataset1_values = [
        [23, 43, 65, 22, 1, 78],
        [34, 23, 45, 65, 23, 43],
        [45, 67, 23, 54, 23, 65],
        [56, 43, 23, 43, 23, 54],
        [67, 65, 34, 65, 12, 43]
    ]

    dataset2_values = [
        [34, 56, 23, 12, 43, 23],
        [45, 34, 56, 54, 23, 54],
        [65, 43, 23, 43, 34, 23],
        [23, 12, 34, 65, 43, 65],
        [43, 23, 65, 34, 23, 54]
    ]

    gene_names = [f"Gene_{i}" for i in range(1, 6)]
    col_names = [f"Cell_{i}" for i in range(1, 7)]

    # Create DataFrames
    dataset1 = pd.DataFrame(dataset1_values, columns=col_names, index=gene_names)
    dataset2 = pd.DataFrame(dataset2_values, columns=col_names, index=gene_names)

    data_list = [dataset1, dataset2]

    mcia_instance = MCIAnalysis(data_list)

    mcia_instance.fit()

    mcia_instance.transform()

    mcia_instance.results()

    tco_result = mcia_instance.column_projection
    # Define the expected result
    expected_result_data = {
        'SV1': [
            1.331175, 0.350811, -0.376169, -0.400608, -0.662327,
            -0.913352, 0.337682, -0.949238, 0.581600, 0.674187
        ],
        'SV2': [
            -0.93431468, 0.88928010, -0.11350972, 0.05502200, 0.09680284,
            -0.82564272, 0.03521497, 0.05176990, 0.78791736, -0.22439781
        ]
    }

    index_values = [
        "Gene_1.Ana1", "Gene_2.Ana1", "Gene_3.Ana1", "Gene_4.Ana1", "Gene_5.Ana1",
        "Gene_1.Ana2", "Gene_2.Ana2", "Gene_3.Ana2", "Gene_4.Ana2", "Gene_5.Ana2",
    ]

    expected_result = pd.DataFrame(expected_result_data, index=index_values)

    assert np.allclose(tco_result, expected_result, atol=1e-6)


def test_mcia_random_datasets_time():
    import time
    start = time.time()

    # Building dataset1 with 1000 observations and positive numbers
    dataset1_values = np.random.randint(0, 50, size=(100, 100)) # Adjust the range as needed
    gene_names = [f"Gene_{i}" for i in range(1, 101)]
    dataset1 = pd.DataFrame(dataset1_values, columns=gene_names)

    # Building dataset2 with 1000 observations and positive numbers
    dataset2_values = np.random.randint(0, 50, size=(100, 100)) # Adjust the range as needed
    dataset2 = pd.DataFrame(dataset2_values, columns=gene_names)

    data_list = [dataset1, dataset2]

    mcia_instance = MCIAnalysis(data_list)

    mcia_instance.fit()

    mcia_instance.transform()

    mcia_instance.results()

    print(time.time() - start)


def test_mcia_pca():
    import time
    start = time.time()

    dataset1_values = np.random.randint(0, 50, size=(150, 100))
    gene_names = [f"Gene_{i}" for i in range(1, 101)]
    dataset1 = pd.DataFrame(dataset1_values, columns=gene_names)

    dataset2_values = np.random.randint(0, 50, size=(150, 100))
    dataset2 = pd.DataFrame(dataset2_values, columns=gene_names)

    data_list = [dataset1, dataset2]

    mcia_instance = MCIAnalysis(data_list)

    mcia_instance.fit(analysis_type = "pca")
    mcia_instance.transform()
    mcia_instance.results()

    elapsed_time = time.time() - start
    print(f"Time elapsed: {elapsed_time} seconds")


def test_mcia_random_datasets_time_2():
    import time
    start = time.time()

    dataset1_values = np.random.randint(0, 50, size=(150, 100))
    gene_names = [f"Gene_{i}" for i in range(1, 101)]
    dataset1 = pd.DataFrame(dataset1_values, columns=gene_names)

    dataset2_values = np.random.randint(0, 50, size=(150, 100))
    dataset2 = pd.DataFrame(dataset2_values, columns=gene_names)

    data_list = [dataset1, dataset2]

    mcia_instance = MCIAnalysis(data_list)

    mcia_instance.fit()
    mcia_instance.transform()
    mcia_instance.results()

    elapsed_time = time.time() - start
    print(f"Time elapsed: {elapsed_time} seconds")



def test_single_dataset():
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
        [67, 34, 56, 54, 43, 23, 65, 34, 56, 65],
    ]

    gene_names = [f"Gene_{i}" for i in range(1, 12)]
    col_names = [f"Cell_{i}" for i in range(1, 11)]

    # Create DataFrames
    dataset1 = pd.DataFrame(dataset1_values, columns=col_names, index=gene_names)

    data_list = [dataset1]

    mcia_instance = MCIAnalysis(data_list)

    mcia_instance.fit()

    mcia_instance.transform()

    mcia_instance.results()

def test_plotting():
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
        [67, 34, 56, 54, 43, 23, 65, 34, 56, 65],
    ]
    gene_names = [f"Gene_{i}" for i in range(1, 12)]
    col_names = [f"Cell_{i}" for i in range(1, 11)]

    # Create DataFrames
    dataset1 = pd.DataFrame(dataset1_values, columns=col_names, index=gene_names)
    dataset2 = pd.DataFrame(dataset2_values, columns=col_names, index=gene_names)

    data_list = [dataset1]

    mcia_instance = MCIAnalysis(data_list)

    mcia_instance.fit()

    mcia_instance.transform()

    mcia_instance.results()

    projected = mcia_instance.project(dataset2)

    import matplotlib.pyplot as plt

    plt.scatter(mcia_instance.column_projection['SV1'], mcia_instance.column_projection['SV2'], c='blue', label='MCIA Data')

    plt.scatter(projected[0], projected[1], s = 122, c='red', label='Protected Data')

    plt.xlabel('SV1')
    plt.ylabel('SV2')
    plt.title('Scatter Plot of MCIA and Protected Data')
    plt.legend()

    plt.show()


def test_plotting_2():
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
        [67, 34, 56, 54, 43, 23, 65, 34, 56, 65],
    ]

    dataset2_values = [
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
        [67, 34, 56, 54, 43, 23, 65, 34, 56, 65],
    ]
    gene_names = [f"Gene_{i}" for i in range(1, 12)]
    col_names = [f"Cell_{i}" for i in range(1, 11)]

    # Create DataFrames
    dataset1 = pd.DataFrame(dataset1_values, columns=col_names, index=gene_names)
    dataset2 = pd.DataFrame(dataset2_values, columns=col_names, index=gene_names)

    data_list = [dataset1]

    mcia_instance = MCIAnalysis(data_list)

    mcia_instance.fit()

    mcia_instance.transform()

    mcia_instance.results()

    projected = mcia_instance.project(dataset2)

    # Create the scatter plot
    plt.scatter(projected[0], projected[1], s = 50, c='red', label='Projected Data')

    plt.scatter(mcia_instance.column_projection['SV1'], mcia_instance.column_projection['SV2'], c='blue', label='MCIA Data')

    plt.xlabel('SV1')
    plt.ylabel('SV2')
    plt.title('Scatter Plot of MCIA and Protected Data')
    plt.legend()

    plt.show()


def test_plotting_3():
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
        [67, 34, 56, 54, 43, 23, 65, 34, 56, 65],
    ]

    dataset2_values = [
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
        [67, 34, 56, 54, 43, 23, 65, 34, 56, 65],
    ]
    gene_names = [f"Gene_{i}" for i in range(1, 12)]
    col_names = [f"Cell_{i}" for i in range(1, 11)]

    # Create DataFrames
    dataset1 = pd.DataFrame(dataset1_values, columns=col_names, index=gene_names)
    dataset2 = pd.DataFrame(dataset2_values, columns=col_names, index=gene_names)

    data_list = [dataset1]

    mcia_instance = MCIAnalysis(data_list)

    mcia_instance.fit()

    mcia_instance.transform()

    mcia_instance.results()

    data_list = [dataset2]

    projected = MCIAnalysis(data_list)

    projected.fit()

    projected.transform()

    projected.results()

    import matplotlib.pyplot as plt

    # Annotate MCIA Data
    for i, txt in enumerate(gene_names):
        plt.annotate(txt, (mcia_instance.column_projection['SV1'][i], mcia_instance.column_projection['SV2'][i]))

    # Annotate Projected Data
    for i, txt in enumerate(gene_names):
        plt.annotate(txt, (projected.column_projection['SV1'][i], projected.column_projection['SV2'][i]))

    plt.scatter(mcia_instance.column_projection['SV1'], mcia_instance.column_projection['SV2'], c='blue', label='MCIA Data')

    plt.scatter(projected.column_projection['SV1'], projected.column_projection['SV2'], s = 15, c='red', label='Protected Data')

    plt.xlabel('SV1')
    plt.ylabel('SV2')
    plt.title('Scatter Plot of MCIA and Protected Data')
    plt.legend()

    plt.show()



def test_plotting_multiple_dataset():
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
        [67, 34, 56, 54, 43, 23, 65, 34, 56, 65],
    ]
    gene_names = [f"Gene_{i}" for i in range(1, 12)]
    col_names = [f"Cell_{i}" for i in range(1, 11)]

    # Create DataFrames
    dataset1 = pd.DataFrame(dataset1_values, columns=col_names, index=gene_names)
    dataset2 = pd.DataFrame(dataset2_values, columns=col_names, index=gene_names)

    data_list = [dataset1, dataset2]

    mcia_instance = MCIAnalysis(data_list)

    mcia_instance.fit()

    mcia_instance.transform()

    mcia_instance.results()

    import matplotlib.pyplot as plt

    num_datasets = 2  # replace N with the number of datasets you want
    chunk_size = mcia_instance.column_projection.shape[0] // num_datasets

    # Initialize list of lists for SV1 and SV2 values
    x = [[] for _ in range(num_datasets)]
    y = [[] for _ in range(num_datasets)]

    # Loop through DataFrame and distribute rows to different datasets
    for i in range(mcia_instance.column_projection.shape[0]):
        dataset_index = i // chunk_size
        if dataset_index >= num_datasets:
            dataset_index = num_datasets - 1  # Put any overflow rows in the last dataset
        x[dataset_index].append(mcia_instance.column_projection.iloc[i]['SV1'])
        y[dataset_index].append(mcia_instance.column_projection.iloc[i]['SV2'])

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x[0], y[0], s=10, c="b", marker="s", label="first")
    ax1.scatter(x[1], y[1], s=10, c="r", marker="o", label="second")
    plt.show()