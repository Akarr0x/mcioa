import sys
import numpy as np
import pandas as pd
import pytest

sys.path.append('/Users/alessandrodiamanti/Desktop/Tesi max planck/Python/mcoia')
from mcoia import mcia
from classes import MCIAnalysis
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

    Tli = mcia_instance.Tli

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

    tl1_result = mcia_instance.Tl1

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

    tco_result = mcia_instance.Tco

    # Define the expected result
    expected_result_data = {
        'SV1': [
            0.97507892, -0.17195575, 0.11365447, -0.38428909, -0.28337742,
            0.39863828, -0.17273745, 0.18178184, 0.22937628, -0.75436653,
            -0.22651066, 0.48788354, -0.68702333, 0.15751381, 0.42787091,
            0.40605912, 0.82405641, -0.45751536, -0.07173074, -0.69153480
        ],
        'SV2': [
            0.34654010, -0.29589539, -0.22224192, 0.46835107, -1.00265768,
            0.55379861, 0.01031573, -0.43030718, -0.02754444, 0.52543316,
            -0.40746059, -0.53827362, 0.08490435, 0.60947838, 0.50412512,
            -0.25158056, -0.23847138, 0.13771407, 0.25748816, -0.21014283
        ]
    }

    index_values = [
        "Gene_1.Ana1", "Gene_2.Ana1", "Gene_3.Ana1", "Gene_4.Ana1", "Gene_5.Ana1",
        "Gene_6.Ana1", "Gene_7.Ana1", "Gene_8.Ana1", "Gene_9.Ana1", "Gene_10.Ana1",
        "Gene_1.Ana2", "Gene_2.Ana2", "Gene_3.Ana2", "Gene_4.Ana2", "Gene_5.Ana2",
        "Gene_6.Ana2", "Gene_7.Ana2", "Gene_8.Ana2", "Gene_9.Ana2", "Gene_10.Ana2"
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

