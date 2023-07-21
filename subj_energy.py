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
C = 0
T = 0.001

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
SC = SCHAEFER['connectivity']
NPARC = int(SC.shape[0])

# LSD Data
LSD_TS = LSD_DATA['ts']
LSD_TS_GSR = LSD_DATA['ts_gsr']
LSD_TS_DIFF = LSD_TS_GSR/LSD_TS
LSD_NSCANS = int(LSD_TS[0, 0].shape[1])
LSD_NSUBJS = int(LSD_TS.shape[0])
LSD_NTRIALS = int(LSD_TS.shape[1])

# NO Data
NO_TS = NO_DATA['ts']
NO_TS_GSR = NO_DATA['ts_gsr']
NO_TS_DIFF = NO_TS_GSR/NO_TS
NO_NSCANS = int(NO_TS[0, 0].shape[1])
NO_NSUBJS = int(NO_TS.shape[0])
NO_NTRIALS = int(NO_TS.shape[1])

# Normalize structural connectivity matrix
ANORM = matrix_normalization(SC, c=C, system='continuous')


def save_energies(ts, nscans, nsubjs, ntrials, saveprefix):
    B = np.eye(NPARC)
    nodal_energy = np.full(ts.shape + (NPARC, nscans-1), np.nan)
    total_energy = np.full(ts.shape + (nscans-1,), np.nan)
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

# save_energies(LSD_TS, LSD_NSCANS, LSD_NSUBJS, LSD_NTRIALS, 'lsd')
# save_energies(NO_TS, NO_NSCANS, NO_NSUBJS, NO_NTRIALS, 'no')
# save_energies(LSD_TS_DIFF, LSD_NSCANS, LSD_NSUBJS, LSD_NTRIALS, 'lsd_diff')
# save_energies(NO_TS_DIFF, NO_NSCANS, NO_NSUBJS, NO_NTRIALS, 'no_diff')

# Load LSD results from save_energies
LSD_NODAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', FOLDER, 'lsd' +
                                        '_nodal_energy.npy'))
LSD_TOTAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', FOLDER, 'lsd' +
                                        '_total_energy.npy'))
LSD_DIFF_NODAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', FOLDER, 'lsd_diff' +
                                             '_nodal_energy.npy'))
LSD_DIFF_TOTAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', FOLDER, 'lsd_diff' +
                                             '_total_energy.npy'))

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
# LSD_TREATMENT_NODAL_AVERAGE = np.mean(
#     [LSD_DIFF_NODAL_ENERGY[:, 0], LSD_DIFF_NODAL_ENERGY[:, 1]], axis=0)
# LSD_SOBER_NODAL_AVERAGE = np.mean(
#     [LSD_DIFF_NODAL_ENERGY[:, 2], LSD_DIFF_NODAL_ENERGY[:, 3]], axis=0)
# regional_ttest(LSD_TREATMENT_NODAL_AVERAGE,
#                LSD_SOBER_NODAL_AVERAGE, 'LSD_diff')

# # Compute and Save NO Regional TTest
# regional_ttest(NO_DIFF_NODAL_ENERGY[:, 0, :],
#                NO_DIFF_NODAL_ENERGY[:, 1, :], 'NO_diff')

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


def regional_plot(regional_t, drug, save=True):
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
    plt.title(
        f"Δ Energy = During Treatment - Before Treatment", size=9)
    plt.suptitle(f"T-Stat for Control Energy of {drug} Condition")
    if save:
        plt.savefig(os.path.join(BASEDIR, 'visuals', FOLDER, drug))
    plt.show()


# regional_plot(LSD_DIFF_REGIONAL_T, 'LSD')
# regional_plot(NO_DIFF_REGIONAL_T, 'NO')
