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
from dominance_analysis import Dominance
import plotly.graph_objects as go
from tabulate import tabulate

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

# Load Regional TTest Results from subj_energy
LSD_T = np.load(os.path.join(
    BASEDIR, 'results', 'subj_energy', 'LSD_diff_regional_t.npy'))
NO_T = np.load(os.path.join(
    BASEDIR, 'results', 'subj_energy', 'NO_diff_regional_t.npy'))

# Dominance analysis
TARGET_NAME = 'regional_t'


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


def run_all_dominance():
    SUBCORTICAL_REGIONS = [i for i in range(101, 117)]
    # All Brain Regions
    LSD_RELATIVE_IMPORTANCE = get_relative_importance(
        LSD_T, 'lsd_relative_importance.csv')
    NO_RELATIVE_IMPORTANCE = get_relative_importance(
        NO_T, 'no_relative_importance.csv')

    # Cortical Brain Regions
    LSD_RELATIVE_IMPORTANCE_CORTICAL = get_relative_importance(
        LSD_T, 'lsd_relative_importance_cortical.csv', regions_exclude=SUBCORTICAL_REGIONS)
    NO_RELATIVE_IMPORTANCE_CORTICAL = get_relative_importance(
        NO_T, 'no_relative_importance_cortical.csv', regions_exclude=SUBCORTICAL_REGIONS)

# run_all_dominance()

# Load the results of run_all_dominance


LSD_RELATIVE_IMPORTANCE = pd.read_csv(os.path.join(
    BASEDIR, 'results', FOLDER, 'lsd_relative_importance.csv'), index_col=0)
NO_RELATIVE_IMPORTANCE = pd.read_csv(os.path.join(
    BASEDIR, 'results', FOLDER, 'no_relative_importance.csv'), index_col=0)

LSD_RELATIVE_IMPORTANCE_CORTICAL = pd.read_csv(os.path.join(
    BASEDIR, 'results', FOLDER, 'lsd_relative_importance_cortical.csv'), index_col=0)
NO_RELATIVE_IMPORTANCE_CORTICAL = pd.read_csv(os.path.join(
    BASEDIR, 'results', FOLDER, 'no_relative_importance_cortical.csv'), index_col=0)


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
    fig.add_trace(scatters['LSD'])
    fig.add_trace(scatters['NO'])
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


SCATTERS_MAIN = {'LSD': create_radial_scatter(
    LSD_RELATIVE_IMPORTANCE, 'LSD'), 'NO': create_radial_scatter(NO_RELATIVE_IMPORTANCE, 'NO')}

SCATTERS_CORTICAL = {'LSD': create_radial_scatter(
    LSD_RELATIVE_IMPORTANCE_CORTICAL, 'LSD'), 'NO': create_radial_scatter(NO_RELATIVE_IMPORTANCE_CORTICAL, 'NO')}


create_radial_plot(SCATTERS_MAIN,
                   'Dominance Analysis for Receptor Mappings', 'dominance_analysis.png')
create_radial_plot(SCATTERS_CORTICAL,
                   'Cortical Dominance Analysis for Receptor Mappings', 'dominance_analysis_cortical.png')
