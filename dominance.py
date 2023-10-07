"""
Must have kaleido to create plotly visual. Run: pip install -U kaleido.

Author: Alec Galin
"""
import numpy as np
import pandas as pd
import scipy.io
import os
from dominance_analysis import Dominance
import plotly.graph_objects as go
from scipy.stats import ttest_rel, zscore
from dominance_analysis import Dominance
import plotly.graph_objects as go
from scipy.stats import ttest_rel, zscore

# Set base directory
BASEDIR = '/Users/alecsmac/coding/coco/drug_comparison'
FOLDER = 'dominance'

# Load receptor data
FIVEHT = scipy.io.loadmat(os.path.join(BASEDIR, 'data', '5HTvecs_sch116.mat'))
RECEPTOR_HEADERS = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                    "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "NMDA", "VAChT"]

RECEPTOR_HEADERS_REDUCED = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                            "CB1", "DAT", "GABAa",
                            "MOR", "NET", "NMDA", "VAChT"]
RECEPTOR_DATA = pd.read_csv(os.path.join(
    BASEDIR, 'data', 'receptor_data_sch116.csv'), names=RECEPTOR_HEADERS)


# REDUCE
RECEPTOR_DATA = RECEPTOR_DATA[RECEPTOR_HEADERS_REDUCED]


# Load relevant LSD results from subj_energy
LSD_DIFF_NODAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', 'subj_energy', 'lsd_diff' +
                                             '_nodal_energy.npy'))
# Get desired LSD result averages
LSD_TREATMENT_NODAL_AVERAGE = np.mean(
    [LSD_DIFF_NODAL_ENERGY[:, 0], LSD_DIFF_NODAL_ENERGY[:, 1]], axis=0)
LSD_SOBER_NODAL_AVERAGE = np.mean(
    [LSD_DIFF_NODAL_ENERGY[:, 2], LSD_DIFF_NODAL_ENERGY[:, 3]], axis=0)

# Load relevant NO results from subj_energy
NO_DIFF_NODAL_ENERGY = np.load(os.path.join(BASEDIR, 'results', 'subj_energy', 'no_diff' +
                                            '_nodal_energy.npy'))


# Compute regional deltas for dominance analysis


# def get_regional_delta(pre, post):
#     def average_transitions(x): return np.mean(x, axis=2)
#     def average_subjects(x): return np.mean(x, axis=0)
#     def average(x): return average_subjects(average_transitions(x))
#     pre, post = average(pre), average(post)
#     return (post-pre)/pre


def get_regional_t(pre, post, zscore_results=False):
    def average_transitions(x): return np.mean(x, axis=2)
    pre_averaged = average_transitions(pre)
    post_averaged = average_transitions(post)
    if zscore_results:
        pre_averaged = zscore(pre_averaged)
        post_averaged = zscore(post_averaged)
    t, pavg = ttest_rel(pre_averaged, post_averaged)
    return t, pavg


# LSD_REGIONAL_DELTA = get_regional_delta(
#     LSD_SOBER_NODAL_AVERAGE, LSD_TREATMENT_NODAL_AVERAGE)
# NO_REGIONAL_DELTA = get_regional_delta(
#     NO_DIFF_NODAL_ENERGY[:, 0], NO_DIFF_NODAL_ENERGY[:, 1])


LSD_T, _ = get_regional_t(LSD_SOBER_NODAL_AVERAGE,
                          LSD_TREATMENT_NODAL_AVERAGE)

LSD_T_ZSCORED, _ = get_regional_t(LSD_SOBER_NODAL_AVERAGE,
                                  LSD_TREATMENT_NODAL_AVERAGE)
NO_T, _ = get_regional_t(
    NO_DIFF_NODAL_ENERGY[:, 0], NO_DIFF_NODAL_ENERGY[:, 1], zscore_results=True)

NO_T_ZSCORED, _ = get_regional_t(
    NO_DIFF_NODAL_ENERGY[:, 0], NO_DIFF_NODAL_ENERGY[:, 1], zscore_results=True)

# Dominance analysis
TARGET_NAME = 'delta_control_energy'


def get_dominance_data(data):
    dominance_data = RECEPTOR_DATA.copy()
    dominance_data[TARGET_NAME] = data
    return dominance_data


TOP_K = 14  # total of 19 receptors but can't run


def dominance_setup(data): return Dominance(
    data=data, target=TARGET_NAME, objective=1, top_k=TOP_K)


def get_relative_importance(regional_delta, title, regions_exclude=[]):
    """
    regions_exclude should be a list of the regions to exclude.
    """
    dominance_data = get_dominance_data(regional_delta)
    if regions_exclude:
        dominance_data = dominance_data.drop([r-1 for r in regions_exclude])
    dominance = dominance_setup(dominance_data)
    # must run this first because of quirk in Dominance class
    dominance.incremental_rsquare()
    relative_importance = dominance.dominance_stats()[
        'Percentage Relative Importance']
    relative_importance.to_csv(os.path.join(BASEDIR, 'results', FOLDER, title))

    return relative_importance


# # Not Z Scored and All Brain Regions
# LSD_RELATIVE_IMPORTANCE = get_relative_importance(
#     LSD_T, 'lsd_relative_importance.csv')
# NO_RELATIVE_IMPORTANCE = get_relative_importance(
#     NO_T, 'no_relative_importance.csv')

# SUBCORTICAL_REGIONS = [i for i in range(101, 117)]

# # Not Z Scored and Cortical Brain Regions
# LSD_RELATIVE_IMPORTANCE_CORTICAL = get_relative_importance(
#     LSD_T, 'lsd_relative_importance_cortical.csv', regions_exclude=SUBCORTICAL_REGIONS)
# NO_RELATIVE_IMPORTANCE__CORTICAL = get_relative_importance(
#     NO_T, 'no_relative_importance_cortical.csv', regions_exclude=SUBCORTICAL_REGIONS)

# # Z Scored and All Brain Regions
# LSD_RELATIVE_IMPORTANCE_ZSCORED = get_relative_importance(
#     LSD_T_ZSCORED, 'lsd_relative_importance_zscored.csv')
# NO_RELATIVE_IMPORTANCE_ZSCORED = get_relative_importance(
#     NO_T_ZSCORED, 'no_relative_importance_zscored.csv')

# # Z Scored and Cortical Brain Regions
# LSD_RELATIVE_IMPORTANCE_CORTICAL_ZSCORED = get_relative_importance(
#     LSD_T_ZSCORED, 'lsd_relative_importance_cortical_zscored.csv', regions_exclude=SUBCORTICAL_REGIONS)
# NO_RELATIVE_IMPORTANCE_CORTICAL_ZSCORED = get_relative_importance(
#     NO_T_ZSCORED, 'no_relative_importance_cortical_zscored.csv', regions_exclude=SUBCORTICAL_REGIONS)

# Load the results generated by the commented code above (takes a while to run)

LSD_RELATIVE_IMPORTANCE = pd.read_csv(os.path.join(
    BASEDIR, 'results', FOLDER, 'lsd_relative_importance.csv'), index_col=0)
NO_RELATIVE_IMPORTANCE = pd.read_csv(os.path.join(
    BASEDIR, 'results', FOLDER, 'no_relative_importance.csv'), index_col=0)

LSD_RELATIVE_IMPORTANCE_CORTICAL = pd.read_csv(os.path.join(
    BASEDIR, 'results', FOLDER, 'lsd_relative_importance_cortical.csv'), index_col=0)
NO_RELATIVE_IMPORTANCE_CORTICAL = pd.read_csv(os.path.join(
    BASEDIR, 'results', FOLDER, 'no_relative_importance_cortical.csv'), index_col=0)

LSD_RELATIVE_IMPORTANCE_ZSCORED = pd.read_csv(os.path.join(
    BASEDIR, 'results', FOLDER, 'lsd_relative_importance_zscored.csv'), index_col=0)
NO_RELATIVE_IMPORTANCE_ZSCORED = pd.read_csv(os.path.join(
    BASEDIR, 'results', FOLDER, 'no_relative_importance_zscored.csv'), index_col=0)

LSD_RELATIVE_IMPORTANCE_CORTICAL_ZSCORED = pd.read_csv(os.path.join(
    BASEDIR, 'results', FOLDER, 'lsd_relative_importance_cortical_zscored.csv'), index_col=0)
NO_RELATIVE_IMPORTANCE_CORTICAL_ZSCORED = pd.read_csv(os.path.join(
    BASEDIR, 'results', FOLDER, 'no_relative_importance_cortical_zscored.csv'), index_col=0)


def create_radial_scatter(data, name):
    """
    Helper for create_radial_plot.
    """
    r = [float(data.loc[receptor]) for receptor in RECEPTOR_HEADERS_REDUCED]
    return go.Scatterpolar(
        r=r,
        theta=RECEPTOR_HEADERS,
        fill='toself',
        name=name
    )


def create_radial_plot(scatters, title, savename):
    fig = go.Figure()
    fig.add_trace(create_radial_scatter(LSD_RELATIVE_IMPORTANCE, 'LSD'))
    fig.add_trace(create_radial_scatter(NO_RELATIVE_IMPORTANCE, 'NO'))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 25],
                tickfont=dict(size=8),
                title='Relative Importance (%)',
                title_font=dict(size=10),
                side='counterclockwise'
            )),
        showlegend=True)
    fig.update_layout(
        title_text=title, title_x=0.5)
    fig.write_image(os.path.join(BASEDIR, 'visuals', FOLDER,
                    savename), scale=3)


SCATTERS_MAIN = [create_radial_scatter(
    LSD_RELATIVE_IMPORTANCE, 'LSD'), create_radial_scatter(NO_RELATIVE_IMPORTANCE, 'NO')]

SCATTERS_CORTICAL = [create_radial_scatter(
    LSD_RELATIVE_IMPORTANCE_CORTICAL, 'LSD'), create_radial_scatter(NO_RELATIVE_IMPORTANCE_CORTICAL, 'NO')]

SCATTERS_MAIN_ZSCORED = [create_radial_scatter(
    LSD_RELATIVE_IMPORTANCE_ZSCORED, 'LSD'), create_radial_scatter(NO_RELATIVE_IMPORTANCE_ZSCORED, 'NO')]

SCATTERS_CORTICAL_ZSCORED = [create_radial_scatter(
    LSD_RELATIVE_IMPORTANCE_CORTICAL_ZSCORED, 'LSD'), create_radial_scatter(NO_RELATIVE_IMPORTANCE_CORTICAL_ZSCORED, 'NO')]


create_radial_plot(SCATTERS_MAIN,
                   'Dominance Analysis for Receptor Mappings', 'dominance_analysis.png')
create_radial_plot(SCATTERS_CORTICAL,
                   'Cortical Dominance Analysis for Receptor Mappings', 'dominance_analysis_cortical.png')
create_radial_plot(SCATTERS_MAIN_ZSCORED,
                   'Z-Scored Dominance Analysis for Receptor Mappings', 'dominance_analysis_zscored.png')
create_radial_plot(SCATTERS_CORTICAL_ZSCORED,
                   'Z-Scored Cortical Dominance Analysis for Receptor Mappings', 'dominance_analysis_cortical_zscored.png')
