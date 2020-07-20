# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:48:43 2020

@author: svc_ccg
"""

from visual_behavior.visualization.extended_trials.daily import make_daily_figure
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.ophys.sync import sync_dataset
from visual_behavior.change_detection.trials.session_metrics import trial_count_by_trial_type
import os
import numpy as np
from matplotlib import pyplot as plt

def get_trials_df(behavior_data):
    
    core_data = data_to_change_detection_core(behavior_data)
    trials = create_extended_dataframe(
        trials=core_data['trials'],
        metadata=core_data['metadata'],
        licks=core_data['licks'],
        time=core_data['time'])
    
    return(trials)


def plot_behavior(trials, save_dir):
    
    daily_behavior_fig = make_daily_figure(trials)
    daily_behavior_fig.savefig(os.path.join(save_dir, 'behavior_summary.png'))
    
    
def get_trial_counts(trials):
    
    trial_counts = []
    labels = []
    for tt in ['hit', 'miss', 'correct_reject', 'false_alarm', 'aborted']:
        trial_counts.append(trial_count_by_trial_type(trials, tt))
        labels.append(tt)
        
    return labels, trial_counts


def plot_trial_type_pie(trial_counts, labels, save_dir):
    
    colors = ['g', '0.5', 'b', 'r', 'orange']
    fig, ax = plt.subplots()
    fig.suptitle('trial types')

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        #return "{:.1f}%\n({:d})".format(pct, absolute)
        return str(absolute)


    wedges, texts, autotexts = ax.pie(trial_counts, colors=colors, 
                                      autopct=lambda pct: func(pct, trial_counts),
                                      textprops=dict(color="w")) 
    ax.legend(wedges, labels,
          title="Trial Types",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
    
    fig.savefig(os.path.join(save_dir, 'trial_type_piechart.png'))
    
    
    
    