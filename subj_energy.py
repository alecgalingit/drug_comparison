"""
Helpful note: In ttest_rel(a,b), the t-statistic is calculated as np.mean(a - b)/se.

Author: Alec Galin
"""
from nctpy.utils import matrix_normalization
from nctpy.energies import minimum_energy_fast
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, zscore
import scipy.io
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm
from brainmontage import create_montage_figure, save_image
import csv
import os

# Set base directory
BASEDIR = '/Users/alecsmac/coding/coco/drug_comparison'
FOLDER = 'subj_energy'

# Set higher resolution for figures
mpl.rcParams['figure.dpi'] = 400

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

# Split LSD results into treatment/sober
LSD_TREATMENT_TOTAL = np.mean(
    [LSD_DIFF_TOTAL_ENERGY[:, 0], LSD_DIFF_TOTAL_ENERGY[:, 1]], axis=0)
LSD_SOBER_TOTAL = np.mean(
    [LSD_DIFF_TOTAL_ENERGY[:, 2], LSD_DIFF_TOTAL_ENERGY[:, 3]], axis=0)
LSD_TREATMENT_NODAL = np.mean(
    [LSD_DIFF_NODAL_ENERGY[:, 0], LSD_DIFF_NODAL_ENERGY[:, 1]], axis=0)
LSD_SOBER_NODAL = np.mean(
    [LSD_DIFF_NODAL_ENERGY[:, 2], LSD_DIFF_NODAL_ENERGY[:, 3]], axis=0)

# Split NO results into treatment/sober
NO_TREATMENT_TOTAL = NO_DIFF_TOTAL_ENERGY[:, 1, :]
NO_SOBER_TOTAL = NO_DIFF_TOTAL_ENERGY[:, 0, :]
NO_TREATMENT_NODAL = NO_DIFF_NODAL_ENERGY[:, 1, :]
NO_SOBER_NODAL = NO_DIFF_NODAL_ENERGY[:, 0, :]

# Average LSD results across transitions for each subject
LSD_TREATMENT_TOTAL_MEAN = np.mean(LSD_TREATMENT_TOTAL, axis=1)
LSD_SOBER_TOTAL_MEAN = np.mean(LSD_SOBER_TOTAL, axis=1)
LSD_TREATMENT_NODAL_MEAN = np.mean(LSD_TREATMENT_NODAL, axis=2)
LSD_SOBER_NODAL_MEAN = np.mean(LSD_SOBER_NODAL, axis=2)

# Average NO results across transitions for each subject
NO_TREATMENT_TOTAL_MEAN = np.mean(NO_TREATMENT_TOTAL, axis=1)
NO_SOBER_TOTAL_MEAN = np.mean(NO_SOBER_TOTAL, axis=1)
NO_TREATMENT_NODAL_MEAN = np.mean(NO_TREATMENT_NODAL, axis=2)
NO_SOBER_NODAL_MEAN = np.mean(NO_SOBER_NODAL, axis=2)

# Save some of these arrays to be used in other scripts
np.save(os.path.join(BASEDIR, 'results', FOLDER,
        'lsd_treatment_total_mean.npy'), LSD_TREATMENT_TOTAL_MEAN)
np.save(os.path.join(BASEDIR, 'results', FOLDER,
        'lsd_sober_total_mean.npy'), LSD_SOBER_TOTAL_MEAN)
np.save(os.path.join(BASEDIR, 'results', FOLDER,
        'no_treatment_total_mean.npy'), NO_TREATMENT_TOTAL_MEAN)
np.save(os.path.join(BASEDIR, 'results', FOLDER,
        'no_sober_total_mean.npy'), NO_SOBER_TOTAL_MEAN)


LSD_GLOBAL_T, LSD_GLOBAL_PAVG = ttest_rel(
    LSD_TREATMENT_TOTAL_MEAN, LSD_SOBER_TOTAL_MEAN)
NO_GLOBAL_T, NO_GLOBAL_PAVG = ttest_rel(
    NO_TREATMENT_TOTAL_MEAN, NO_SOBER_TOTAL_MEAN)
print('LSD', LSD_GLOBAL_T, LSD_GLOBAL_PAVG)
print('NO', NO_GLOBAL_T, NO_GLOBAL_PAVG)

# Regional Energy T-Tests


def regional_ttest(data_after, data_before, saveprefix):
    postfix = ''
    t, pavg = ttest_rel(data_after, data_before)
    np.save(os.path.join(BASEDIR, 'results', FOLDER,
            f'{saveprefix}_regional_t{postfix}.npy'), t)
    np.save(os.path.join(BASEDIR, 'results', FOLDER,
            f'{saveprefix}_regional_pavg{postfix}.npy'), pavg)
    return t, pavg


# Compute and Save LSD Regional TTests
regional_ttest(LSD_TREATMENT_NODAL_MEAN, LSD_SOBER_NODAL_MEAN, 'LSD_diff')

# Compute and Save NO Regional TTests
regional_ttest(NO_TREATMENT_NODAL_MEAN, NO_SOBER_NODAL_MEAN, 'NO_diff')

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


def regional_plot(regional_t, title, savename, subtitle='test', save=True):
    range = [np.min(regional_t), np.max(regional_t)]
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
    # max/mins
    fig.text(0.5, 1.045, subtitle,
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes, style='italic', size=8)
    fig.text(0.73, 0.5, 'T-Statistic')
    # plt.title(
    #     subtitle, size=9)
    plt.suptitle(title)
    if save:
        plt.savefig(os.path.join(BASEDIR, 'visuals',
                    FOLDER, savename))
    # plt.show()


# Uncomment to visualize regional t test results
regional_plot(LSD_DIFF_REGIONAL_T, "Change in Control Energy for LSD Subjects", 'regional_tstat_LSD',
              "A negative t-stat indicates a decrease in control energy after treatment")
regional_plot(NO_DIFF_REGIONAL_T, "Change in Control Energy for NO Subjects", 'regional_tstat_NO',
              "A negative t-stat indicates a decrease in control energy after treatment")

# Global Delta Energy Unpaired t test


def delta_energy(sober, high):
    return (high-sober) / sober


DELTA_ENERGY_NO = delta_energy(NO_SOBER_TOTAL_MEAN, NO_TREATMENT_TOTAL_MEAN)
DELTA_ENERGY_LSD = delta_energy(LSD_SOBER_TOTAL_MEAN, LSD_TREATMENT_TOTAL_MEAN)


DELTA_ENERGY_T, DELTA_ENERGY_PAVG = ttest_ind(DELTA_ENERGY_LSD,
                                              DELTA_ENERGY_NO, equal_var=False)


def print_global_delta_t():
    print('\n')
    print(f'Global T for Delta Energy: {DELTA_ENERGY_T}')
    print(f'Global PAVG for Delta Energy: {DELTA_ENERGY_PAVG}')
    print(f'A Negative T stat indicates that there is a greater decrease in energy for LSD than for NO.')
    print('\n')


print_global_delta_t()

# Regional Delta Energys Unpaired ttest

REGIONAL_DELTA_ENERGY_LSD = delta_energy(
    LSD_SOBER_NODAL_MEAN, LSD_TREATMENT_NODAL_MEAN)
REGIONAL_DELTA_ENERGY_NO = delta_energy(
    NO_SOBER_NODAL_MEAN, NO_TREATMENT_NODAL_MEAN)

REGIONAL_DELTA_ENERGY_T, REGIONAL_DELTA_ENERGY_PAVG = ttest_ind(REGIONAL_DELTA_ENERGY_LSD,
                                                                REGIONAL_DELTA_ENERGY_NO, equal_var=False)


regional_plot(REGIONAL_DELTA_ENERGY_T,
              'Difference in ΔEnergy Between LSD and NO Conditions', 'delta_control_energy_tstat', "A negative t-stat indicates a greater decrease in control energy after treatment for LSD subjects")
