"""
Author: Alec Galin
"""
from nctpy.utils import matrix_normalization
from nctpy.energies import minimum_energy_fast
import numpy as np
from scipy.stats import ttest_rel, ttest_ind
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm
from matplotlib.cm import coolwarm
from brainmontage import create_montage_figure, save_image
import seaborn as sns
import csv
import os

# Set base directory
BASEDIR = '/Users/alecsmac/coding/coco/drug_comparison'
FOLDER = 'subj_energy'

# Set variables for energy calculations
C = 1
T = 1

# Load Data
LSD_DATA = scipy.io.loadmat(os.path.join(
    BASEDIR, 'data', 'LSD_clean_mni_nomusic_sch116.mat'))
NO_DATA = scipy.io.loadmat(os.path.join(
    BASEDIR, 'data', 'NOx_clean_mni_sch116.mat'))
SCHAEFER = scipy.io.loadmat(os.path.join(
    BASEDIR, "data", "Schaefer116_HCP_DTI_count.mat"))
YEO_FILE = open(os.path.join(BASEDIR, "data", "sch116_to_yeo.csv"))
YEO = [int(i) for i in list(csv.reader(YEO_FILE, delimiter=","))[0]]
YEO_FILE.close()
NETWORKS = ['VIS', 'SOM', 'DAT', 'VAT', 'LIM', 'FPN', 'DMN', 'SUB']

# Extract relevant variables
SC = SCHAEFER['vol_normalized_sc']
NPARC = int(SC.shape[0])


def convert_to_one_array(array_of_arrays):
    return np.stack([np.stack(array_of_arrays[i], axis=0)
                     for i in range(array_of_arrays.shape[0])], axis=0)


def get_ts_diff(ts, ts_gsr):
    ts_diff = ts_gsr.copy()
    for i1 in range(ts.shape[0]):
        for i2 in range(ts.shape[1]):
            for i3 in range(ts.shape[2]):
                ts_mean = np.mean(ts[i1, i2, i3])
                ts_diff[i1, i2, i3] = ts_diff[i1, i2, i3]/ts_mean
    return ts_diff


# LSD Data
LSD_TS = convert_to_one_array(LSD_DATA['ts'])
LSD_TS_GSR = convert_to_one_array(LSD_DATA['ts_gsr'])
LSD_TS_DIFF = get_ts_diff(LSD_TS, LSD_TS_GSR)
LSD_NSCANS = int(LSD_TS.shape[3])
LSD_NSUBJS = int(LSD_TS.shape[0])
LSD_NTRIALS = int(LSD_TS.shape[1])
# NO Data
NO_TS = convert_to_one_array(NO_DATA['ts'])
NO_TS_GSR = convert_to_one_array(NO_DATA['ts_gsr'])
NO_TS_DIFF = NO_TS_GSR  # get_ts_diff(NO_TS, NO_TS_GSR)
NO_NSCANS = int(NO_TS.shape[3])
NO_NSUBJS = int(NO_TS.shape[0])
NO_NTRIALS = int(NO_TS.shape[1])

# Normalize structural connectivity matrix
ANORM = matrix_normalization(SC, c=C, system='continuous')


def save_energies(ts, nscans, nsubjs, ntrials, saveprefix):
    B = np.eye(NPARC)
    nodal_energy = np.full(ts.shape[:3] + (nscans-1,), np.nan)
    total_energy = np.full(ts.shape[:2] + (nscans-1,), np.nan)
    ts_vec = np.arange(0, nscans)
    xi_index = ts_vec[:-1]
    xf_index = ts_vec[1:]
    for subj in range(nsubjs):
        for trial in range(ntrials):
            xi = ts[subj, trial].T[xi_index]
            xf = ts[subj, trial].T[xf_index]
            energies = minimum_energy_fast(
                A_norm=ANORM, T=T, B=B, x0=xi.T, xf=xf.T)
            tot_energies = np.sum(energies, axis=0)
            nodal_energy[subj, trial, :] = energies
            total_energy[subj, trial, :] = tot_energies
    np.save(os.path.join(BASEDIR, 'results', FOLDER, saveprefix +
            '_nodal_energy.npy'), nodal_energy)
    np.save(os.path.join(BASEDIR, 'results', FOLDER, saveprefix +
            '_total_energy.npy'), total_energy)

    return nodal_energy, total_energy


# Calculate and save control energies

save_energies(LSD_TS, LSD_NSCANS, LSD_NSUBJS, LSD_NTRIALS, 'lsd')
save_energies(NO_TS, NO_NSCANS, NO_NSUBJS, NO_NTRIALS, 'no')
save_energies(LSD_TS_DIFF, LSD_NSCANS, LSD_NSUBJS, LSD_NTRIALS, 'lsd_diff')
save_energies(NO_TS_DIFF, NO_NSCANS, NO_NSUBJS, NO_NTRIALS, 'no_diff')

# Load LSD results from save_energies
LSD_NODAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', FOLDER, 'lsd' +
                                        '_nodal_energy.npy'))
LSD_TOTAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', FOLDER, 'lsd' +
                                        '_total_energy.npy'))
LSD_DIFF_NODAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', FOLDER, 'lsd_diff' +
                                             '_nodal_energy.npy'))
LSD_DIFF_TOTAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', FOLDER, 'lsd_diff' +
                                             '_total_energy.npy'))
# Get desired LSD result averages
LSD_TREATMENT_TOTAL_AVERAGE = np.mean(
    [LSD_DIFF_TOTAL_ENERGY[:, 0], LSD_DIFF_TOTAL_ENERGY[:, 1]], axis=0)
LSD_SOBER_TOTAL_AVERAGE = np.mean(
    [LSD_DIFF_TOTAL_ENERGY[:, 2], LSD_DIFF_TOTAL_ENERGY[:, 3]], axis=0)
LSD_TREATMENT_NODAL_AVERAGE = np.mean(
    [LSD_DIFF_NODAL_ENERGY[:, 0], LSD_DIFF_NODAL_ENERGY[:, 1]], axis=0)
LSD_SOBER_NODAL_AVERAGE = np.mean(
    [LSD_DIFF_NODAL_ENERGY[:, 2], LSD_DIFF_NODAL_ENERGY[:, 3]], axis=0)

# Load NO results from save_energies
NO_NODAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', FOLDER, 'no' +
                                       '_nodal_energy.npy'))
NO_TOTAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', FOLDER, 'no' +
                                       '_total_energy.npy'))
NO_DIFF_NODAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', FOLDER, 'no_diff' +
                                            '_nodal_energy.npy'))
NO_DIFF_TOTAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', FOLDER, 'no_diff' +
                                            '_total_energy.npy'))


# Global Energy T-Tests

def LSD_global_energy_ttest():
    lsd_condition_average = np.mean([LSD_DIFF_TOTAL_ENERGY[:, 0, :],
                                     LSD_DIFF_TOTAL_ENERGY[:, 1, :]], axis=0)
    placebo_condition_average = np.mean([LSD_DIFF_TOTAL_ENERGY[:, 2, :],
                                         LSD_DIFF_TOTAL_ENERGY[:, 3, :]], axis=0)
    t, pavg = ttest_rel(
        np.mean(lsd_condition_average, axis=1), np.mean(placebo_condition_average, axis=1))
    return t, pavg


def NO_global_energy_ttest():
    t, pavg = ttest_rel(
        np.mean(NO_DIFF_TOTAL_ENERGY[:, 0, :], axis=1), np.mean(
            NO_DIFF_TOTAL_ENERGY[:, 1, :], axis=1))
    return t, pavg

# Regional Energy T-Tests


def regional_ttest(data_before, data_after, saveprefix):
    before_average = np.mean(data_before, axis=2)
    after_average = np.mean(data_after, axis=2)
    t, pavg = ttest_rel(before_average, after_average)
    np.save(os.path.join(BASEDIR, 'results', FOLDER, saveprefix +
            '_regional_t.npy'), t)
    np.save(os.path.join(BASEDIR, 'results', FOLDER, saveprefix +
            '_regional_pavg.npy'), pavg)
    return t, pavg


# Compute and Save LSD Regional TTest
regional_ttest(LSD_TREATMENT_NODAL_AVERAGE,
               LSD_SOBER_NODAL_AVERAGE, 'LSD_diff')

# Compute and Save NO Regional TTest
regional_ttest(NO_DIFF_NODAL_ENERGY[:, 0, :],
               NO_DIFF_NODAL_ENERGY[:, 1, :], 'NO_diff')

# Load Regional TTest Results
LSD_DIFF_REGIONAL_T = np.load(os.path.join(
    BASEDIR, 'results', FOLDER, 'LSD_diff_regional_t.npy'))
LSD_DIFF_REGIONAL_PAVG = np.load(os.path.join(
    BASEDIR, 'results', FOLDER, 'LSD_diff_regional_pavg.npy'))
NO_DIFF_REGIONAL_T = np.load(os.path.join(
    BASEDIR, 'results', FOLDER, 'NO_diff_regional_t.npy'))
NO_DIFF_REGIONAL_PAVG = np.load(os.path.join(
    BASEDIR, 'results', FOLDER, 'NO_diff_regional_pavg.npy'))

# Visualize Regional TTest Results


def regional_plot(regional_t, title, savename, subtitle='', save=True):
    range = [np.min(regional_t), np.max(regional_t)]
    tot_range = abs(range[1] - range[0])
    zero_distance = abs(range[0])
    img = create_montage_figure(regional_t, atlasname='schaefer100', viewnames='all',
                                surftype='infl', clim=[-5, 5], colormap=coolwarm, add_colorbar=True)
    fig = plt.figure()
    plt.imshow(img, )
    plt.axis('off')
    ax = fig.gca()
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])
    # plt.title(
    #     subtitle, size=9)
    plt.suptitle(title)
    if save:
        plt.savefig(os.path.join(BASEDIR, 'visuals', FOLDER, savename))
    plt.show()


# Uncomment to visualize regional t test results
regional_plot(LSD_DIFF_REGIONAL_T, "T-Stat for Control Energy of LSD Condition", 'regional_tstat_LSD',
              "Δ Energy = During Treatment - Before Treatment")
regional_plot(NO_DIFF_REGIONAL_T, "T-Stat for Control Energy of NO Condition", 'regional_tstat_NO'
              "Δ Energy = During Treatment - Before Treatment")

# Global Delta Energy Unpaired t test


def delta_energy(sober, high):
    return (sober - high) / sober


def delta_energy_NO():
    mean_across_transitions = np.mean(NO_DIFF_TOTAL_ENERGY, axis=2)
    sober = mean_across_transitions[:, 0]
    high = mean_across_transitions[:, 1]
    return delta_energy(sober, high)


def delta_energy_LSD():
    sober = np.mean(LSD_SOBER_TOTAL_AVERAGE, axis=1)
    high = np.mean(LSD_TREATMENT_TOTAL_AVERAGE, axis=1)
    return delta_energy(sober, high)


DELTA_ENERGY_NO = delta_energy_NO()
DELTA_ENERGY_LSD = delta_energy_LSD()


DELTA_ENERGY_T, DELTA_ENERGY_PAVG = ttest_ind(DELTA_ENERGY_LSD,
                                              DELTA_ENERGY_NO, equal_var=False)


def print_global_delta_t():
    print('\n')
    print(f'Global T for Delta Energy: {DELTA_ENERGY_T}')
    print(f'Global PAVG for Delta Energy: {DELTA_ENERGY_PAVG}')
    print('\n')


print_global_delta_t()

# Regional Delta Energys Unpaired ttest


def regional_delta_energy_NO():
    mean_across_transitions = np.mean(NO_DIFF_NODAL_ENERGY, axis=3)
    sober = mean_across_transitions[:, 0, :]
    high = mean_across_transitions[:, 1, :]
    return delta_energy(sober, high)


def regional_delta_energy_LSD():
    sober = np.mean(LSD_SOBER_NODAL_AVERAGE, axis=2)
    high = np.mean(LSD_TREATMENT_NODAL_AVERAGE, axis=2)
    return delta_energy(sober, high)


REGIONAL_DELTA_ENERGY_NO = regional_delta_energy_NO()
REGIONAL_DELTA_ENERGY_LSD = regional_delta_energy_LSD()

REGIONAL_DELTA_ENERGY_T, REGIONAL_DELTA_ENERGY_PAVG = ttest_ind(REGIONAL_DELTA_ENERGY_LSD,
                                                                REGIONAL_DELTA_ENERGY_NO, equal_var=False)


regional_plot(REGIONAL_DELTA_ENERGY_T,
              'Unpaired T-Test for Δ Control Energy of LSD Vs. NO Conditions', 'delta_control_energy_tstat')
