import numpy as np
import pandas as pd
import scipy.io
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set base directory
BASEDIR = '/Users/alecsmac/coding/coco/drug_comparison'
FOLDER = 'displacement'

# Load Data
LSD_DATA = scipy.io.loadmat(os.path.join(
    BASEDIR, 'data', 'LSD_clean_mni_nomusic_sch116.mat'))
NO_DATA = scipy.io.loadmat(os.path.join(
    BASEDIR, 'data', 'NOx_clean_mni_sch116.mat'))

print(LSD_DATA['__globals__'])
print(NO_DATA['__globals__'])

# Load relevant LSD results from subj_energy
LSD_DIFF_TOTAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', 'subj_energy', 'lsd_diff' +
                                             '_total_energy.npy'))
# # Get desired LSD result averages
LSD_TREATMENT_NODAL_AVERAGE = np.mean(
    [LSD_DIFF_TOTAL_ENERGY[:, 0], LSD_DIFF_TOTAL_ENERGY[:, 1]], axis=0)
LSD_SOBER_NODAL_AVERAGE = np.mean(
    [LSD_DIFF_TOTAL_ENERGY[:, 2], LSD_DIFF_TOTAL_ENERGY[:, 3]], axis=0)

# Load relevant NO results from subj_energy
NO_DIFF_TOTAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', 'subj_energy', 'no_diff' +
                                            '_total_energy.npy'))


def pc_scatterplot_bids():
    pc_data_fixed = pc_data.copy()
    pc_data_fixed['Arm'] = pc_data_fixed['Arm'].replace(
        {0.: 'SSRI', 1.: 'Psilocybin'})
    pc_data_fixed['Sex'] = pc_data_fixed['Sex'].replace(
        {0.: 'Male', 1.: 'Female'})
    pc_data_fixed['Discontinued'] = pc_data_fixed['Discontinued'].replace(
        {0.: False, 1.: True})
    sns.set_theme()
    sns.set_style("white")
    rel = sns.relplot(data=pc_data_fixed, x='energy_change', y='bids_change',
                      hue='Arm', palette='rocket', style='Discontinued')
    rel.fig.suptitle(
        "Relative Change in Control Energy vs Relative Change in BIDS Scores")
    plt.gcf().subplots_adjust(bottom=0.25, top=0.85)
    rel.set_axis_labels("Relative Change in Control Energy",
                        "Relative Change in BIDS Score")
    rel.fig.set_size_inches(11, 5.5)
    rel.fig.text(0.3, 0.07, "Pearson Partial Correlation", fontweight='bold',
                 ha='center', va='bottom')
    rel.fig.text(0.7, 0.07, "Spearman Partial Correlation", fontweight='bold',
                 ha='center', va='bottom')
    rel.fig.text(0.3, 0, pc_pearson_bids.to_string(),
                 ha='center', va='bottom')
    rel.fig.text(0.7, 0, pc_spearman_bids.to_string(),
                 ha='center', va='bottom')
    plt.savefig(
        f'{basedir}/Plots/{folder}/parco_bids.png', dpi=300)
    plt.show()
