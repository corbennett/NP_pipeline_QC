# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:15:39 2023

@author: svc_ccg
"""
import pandas as pd
from behavior_analysis import *
import numpy as np


p = pd.read_pickle(r"\\allen\programs\mindscope\workgroups\dynamicrouting\novel_midway_script\646316\230405172107.pkl")
trial_log = p['items']['behavior']['trial_log']
trials = get_trials_df(p)

previous_image_set = trial_log[0]['image_set']
switch_trials = []
for it, trial in enumerate(trial_log):
    if it==0:
        continue
    
    image_set = trial['image_set']
    if image_set != previous_image_set:
        switch_trials.append(it)
    
    previous_image_set = image_set

smooth_factor = 10
for responsetype in ['EARLY_RESPONSE', 'FA', 'HIT', 'MISS', 'CR']:
    
    fig, ax = plt.subplots()
    fig.suptitle(responsetype)
    
    if responsetype != 'EARLY_RESPONSE':
        trials_to_use = trials[trials['response_type']!='EARLY_RESPONSE'].index.values
        switches = np.searchsorted(trials_to_use, switch_trials)
    else:
        trials_to_use = trials.index.values
        switches = np.copy(switch_trials)
    
    responsevector = trials.loc[trials_to_use]['response_type']==responsetype
    smoothed = np.convolve(responsevector, np.ones(smooth_factor))/smooth_factor
    
    ax.plot(smoothed)
    [ax.axvline(switch, color='k', linestyle='dotted') for switch in switches]