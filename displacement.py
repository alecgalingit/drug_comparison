"""
Openpyxl is a required dependency.
"""
import numpy as np
import pandas as pd
import scipy.io
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Set base directory
BASEDIR = '/Users/alecsmac/coding/coco/drug_comparison'
FOLDER = 'displacement'

# Set higher resolution for figures
mpl.rcParams['figure.dpi'] = 400


LSD_TREATMENT_TOTAL_MEAN = np.load(os.path.join(BASEDIR, 'results', 'subj_energy', 'lsd_treatment_total_mean.npy'))
LSD_SOBER_TOTAL_MEAN = np.load(os.path.join(BASEDIR, 'results', 'subj_energy', 'lsd_sober_total_mean.npy'))
NO_TREATMENT_TOTAL_MEAN = np.load(os.path.join(BASEDIR, 'results', 'subj_energy', 'no_treatment_total_mean.npy'))
NO_SOBER_TOTAL_MEAN = np.load(os.path.join(BASEDIR, 'results', 'subj_energy', 'no_sober_total_mean.npy'))

# Load framewise displacement data
LSD_FD = pd.read_excel(os.path.join(
    BASEDIR, 'data', 'Ratings_and_motion.xlsx'))
NO_FD = pd.read_excel(os.path.join(BASEDIR, 'data', 'DEM+FD.xlsx'))

# Fix LSD column names


def fix_LSD_fd_columns(lsd_fd):
    lsd_fd = lsd_fd.copy()
    new_columns = list(lsd_fd.columns[:2]) + list(lsd_fd.iloc[0, 2:])
    lsd_fd.columns = new_columns
    lsd_fd = lsd_fd.drop(0)
    return lsd_fd


LSD_FD = fix_LSD_fd_columns(LSD_FD)


def create_fd_data_for_drug(baseline_fd, high_fd, baseline_te, high_te, condition):
    to_plot = pd.DataFrame(
        columns=['Mean FD', 'Total Energy', 'Treatment', 'Condition'])
    to_plot['Global TE'] = np.concatenate([baseline_te, high_te])
    to_plot['Mean FD'] = pd.concat(
        [baseline_fd, high_fd], axis=0, ignore_index=True)
    treatment = [False] * baseline_te.shape[0] + [True] * high_te.shape[0]
    drug = [condition]*(baseline_te.shape[0] + high_te.shape[0])
    to_plot['Treatment'] = pd.Series(treatment)
    to_plot['Condition'] = pd.Series(drug)
    return to_plot


def create_fd_data():
    # get and add NO data
    no_baseline_displacement = NO_FD['meanFD(baseline)']
    no_high_displacement = NO_FD['meanFD(N2O)']
    no_baseline_TE = NO_SOBER_TOTAL_MEAN
    no_high_TE = NO_TREATMENT_TOTAL_MEAN

    to_plot_no = create_fd_data_for_drug(no_baseline_displacement,
                                         no_high_displacement, no_baseline_TE, no_high_TE, 'NO')

    # get and add LSD data
    lsd_baseline_displacement = LSD_FD.iloc[:15, 9]
    lsd_high_displacement = LSD_FD.iloc[:15, 13]
    lsd_baseline_TE = LSD_SOBER_TOTAL_MEAN
    lsd_high_TE = LSD_TREATMENT_TOTAL_MEAN

    to_plot_lsd = create_fd_data_for_drug(lsd_baseline_displacement,
                                          lsd_high_displacement, lsd_baseline_TE, lsd_high_TE, 'LSD')
    all_to_plots = [to_plot_no, to_plot_lsd]
    return pd.concat(all_to_plots, ignore_index=True)


def meanfdplot_for_drug(data,drug):
    sns.set_theme()
    sns.set_style("white")
    rel = sns.relplot(data=data, x='Mean FD', y='Global TE',
                      hue='Treatment', palette='rocket')
    rel.fig.suptitle(
        f"Mean FD vs Global TE for {drug}")
    plt.gcf().subplots_adjust(bottom=0.25, top=0.85)
    rel.set_axis_labels("Mean Framewise Displacement",
                        "Global Transition Energy")
    rel.fig.set_size_inches(11, 5.5)
    plt.savefig(os.path.join(BASEDIR, 'visuals', FOLDER,
                    f'meanfd_{drug}.png'))
    plt.close()


def all_meanfdplots():
    data = create_fd_data()
    lsd_data = data[data['Condition']=='LSD']
    no_data = data[data['Condition']=='NO']
    meanfdplot_for_drug(lsd_data,'LSD')
    meanfdplot_for_drug(no_data,'NO')

all_meanfdplots()
