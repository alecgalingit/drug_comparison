import numpy as np
import pandas as pd
import scipy.io
import os
import matplotlib.pyplot as plt

BASEDIR = '/Users/alecsmac/coding/coco/drug_comparison'
FOLDER = 'receptor_densities'

FIVEHT = scipy.io.loadmat(os.path.join(BASEDIR, 'data', '5HTvecs_sch116.mat'))
RECEPTOR_HEADERS = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                    "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "NMDA", "VAChT"]


RECEPTOR_DATA = pd.read_csv(os.path.join(
    BASEDIR, 'data', 'receptor_data_sch116.csv'), names=RECEPTOR_HEADERS)


def create_histograms(receptor_data):
    for column in receptor_data.columns:
        plt.figure()
        receptor_data[column].hist()
        plt.title(f'Receptor Density Histogram for {column}')
        plt.xlabel('Receptor Density')
        plt.ylabel('Frequency of Brain Regions')

        plt.savefig(os.path.join(BASEDIR, 'visuals', FOLDER,
                    f'{column}_density_histogram.png'))
        plt.close()


# create_histograms(RECEPTOR_DATA)
print(RECEPTOR_DATA['5HT1a'])
