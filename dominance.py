"""
Must have kaleido to create plotly visual. Run: pip install -U kaleido.

Author: Alec
"""
import numpy as np
import pandas as pd
import scipy.io
import os
from dominance_analysis import Dominance
from dominance_analysis import Dominance_Datasets
import re
import plotly.graph_objects as go

# Set base directory
BASEDIR = '/Users/alecsmac/coding/coco/drug_comparison'
FOLDER = 'dominance'

# Load receptor data
FIVEHT = scipy.io.loadmat(os.path.join(BASEDIR, 'data', '5HTvecs_sch116.mat'))

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
def get_regional_delta(pre, post):
    average_transitions = lambda x : np.mean(x,axis=2)
    average_subjects = lambda x : np.mean(x,axis=0)
    average = lambda x : average_subjects(average_transitions(x))
    pre, post = average(pre), average(post)
    return (post-pre)/pre

LSD_REGIONAL_DELTA = get_regional_delta(LSD_SOBER_NODAL_AVERAGE, LSD_TREATMENT_NODAL_AVERAGE)
NO_REGIONAL_DELTA = get_regional_delta(NO_DIFF_NODAL_ENERGY[:,0], NO_DIFF_NODAL_ENERGY[:,1])

# Dominance analysis

def get_receptor_name_mappings():
  output = {}
  for k in FIVEHT.keys():
      match = re.search(r'mean(.*)_sch116', k)
      if match:
         output[k] = match.group(1)
  return output

RECEPTOR_NAME_MAPPINGS  = get_receptor_name_mappings()
TARGET_NAME = 'delta_control_energy'

def get_dominance_data(data):
  receptor_data_list = [np.squeeze(FIVEHT[receptor]) for receptor in RECEPTOR_NAME_MAPPINGS]
  together = (data,) + tuple(receptor_data_list)
  data_list = np.column_stack(together)
  return pd.DataFrame(data_list, columns=[TARGET_NAME] + list(RECEPTOR_NAME_MAPPINGS.keys()))


LSD_DOMINANCE_DATA = get_dominance_data(LSD_REGIONAL_DELTA)
NO_DOMINANCE_DATA = get_dominance_data(NO_REGIONAL_DELTA)

dominance_setup = lambda data : Dominance(data=data,target=TARGET_NAME,objective=1)

LSD_DOMINANCE = dominance_setup(LSD_DOMINANCE_DATA)
NO_DOMINANCE = dominance_setup(NO_DOMINANCE_DATA)

def get_relative_importance(dominance):
   dominance.incremental_rsquare() # must run this first because of quirk in Dominance class
   return dominance.dominance_stats()['Percentage Relative Importance']

LSD_RELATIVE_IMPORTANCE = get_relative_importance(LSD_DOMINANCE)
NO_RELATIVE_IMPORTANCE = get_relative_importance(NO_DOMINANCE)

def create_radial_scatter(data, name):
   """
   Helper for create_radial_plot.
   """
   r = [float(data.loc[receptor]) for receptor in RECEPTOR_NAME_MAPPINGS]
   categories = list(RECEPTOR_NAME_MAPPINGS.values())
   return go.Scatterpolar(
      r=r,
      theta=categories,
      fill='toself',
      name=name
    )

def create_radial_plot():
   categories = RECEPTOR_NAME_MAPPINGS.values()
   fig = go.Figure()
   fig.add_trace(create_radial_scatter(LSD_RELATIVE_IMPORTANCE, 'LSD'))
   fig.add_trace(create_radial_scatter(NO_RELATIVE_IMPORTANCE, 'NO'))
   fig.update_layout(
    polar=dict(
    radialaxis=dict(
    visible=True,
    range=[0, 60],
    tickfont=dict(size=8),
    title='Relative Importance (%)',
    title_font=dict(size=10),
    side = 'counterclockwise'
    )),
  showlegend=True)
   fig.update_layout(title_text='Dominance Analysis for Receptor Mappings', title_x=0.5)
   fig.write_image(os.path.join(BASEDIR, 'visuals', FOLDER, 'dominance_analysis.png'), scale=2)

create_radial_plot()
