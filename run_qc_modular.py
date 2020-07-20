# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:42:31 2020

@author: svc_ccg
"""

import numpy as np
import os, glob, shutil
import behavior_analysis
from visual_behavior.ophys.sync import sync_dataset
import pandas as pd
from matplotlib import pyplot as plt
import analysis
import probeSync_qc as probeSync
import scipy.signal
import cv2
import data_getters
from get_RFs_standalone import get_RFs

import logging 

### SPECIFY EXPERIMENT TO PULL ####
#this should be either the ten digit lims id:
#identifier = '1013651431' #lims id
#or the local base directory
identifier = r'\\10.128.50.43\sd6.3\1033616558_509940_20200701'
identifier = r'\\10.128.50.43\sd6.3\1033388795_509652_20200630' 
identifier = r"\\10.128.50.43\sd6.3\1033387557_509940_20200630"


if identifier.find('_')>=0:
    d = data_getters.local_data_getter(base_dir=identifier)
else:
    d = data_getters.lims_data_getter(exp_id=identifier)

paths = d.data_dict
FIG_SAVE_DIR = os.path.join(r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\QC", 
                            paths['es_id']+'_'+paths['external_specimen_name']+'_'+paths['datestring'])
if not os.path.exists(FIG_SAVE_DIR):
    os.mkdir(FIG_SAVE_DIR)

### GET FILE PATHS TO SYNC AND PKL FILES ###
SYNC_FILE = paths['sync_file']
BEHAVIOR_PKL = paths['behavior_pkl']
REPLAY_PKL = paths['replay_pkl']
MAPPING_PKL = paths['mapping_pkl']

for f,s in zip([SYNC_FILE, BEHAVIOR_PKL, REPLAY_PKL, MAPPING_PKL], ['sync: ', 'behavior: ', 'replay: ', 'mapping: ']):
    print(s + f)
    

### GET MAIN DATA STREAMS ###
syncDataset = sync_dataset.Dataset(SYNC_FILE)
behavior_data = pd.read_pickle(BEHAVIOR_PKL)
mapping_data = pd.read_pickle(MAPPING_PKL)
replay_data = pd.read_pickle(REPLAY_PKL)

### Behavior Analysis ###
trials = behavior_analysis.get_trials_df(behavior_data)
behavior_analysis.plot_behavior(trials, FIG_SAVE_DIR)

trial_types, counts = behavior_analysis.get_trial_counts(trials)
behavior_analysis.plot_trial_type_pie(counts, trial_types, FIG_SAVE_DIR)


### CHECK FRAME COUNTS ###
vr, vf = probeSync.get_sync_line_data(syncDataset, channel=2)


    
behavior_frame_count = behavior_data['items']['behavior']['intervalsms'].size + 1
mapping_frame_count = mapping_data['intervalsms'].size + 1
replay_frame_count = replay_data['intervalsms'].size + 1

total_pkl_frames = (behavior_frame_count +
                    mapping_frame_count +
                    replay_frame_count) 

### CHECK THAT NO FRAMES WERE DROPPED FROM SYNC ###
print('frames in pkl files: {}'.format(total_pkl_frames))
print('frames in sync file: {}'.format(len(vf)))

assert(total_pkl_frames==len(vf))

### CHECK THAT REPLAY AND BEHAVIOR HAVE SAME FRAME COUNT ###
print('frames in behavior stim: {}'.format(behavior_frame_count))
print('frames in replay stim: {}'.format(replay_frame_count))

assert(behavior_frame_count==replay_frame_count)

# look for potential frame offsets from aborted stims
stimstarts, stimoffs = probeSync.get_sync_line_data(syncDataset, 'stim_running')
if len(stimstarts)>3:
    logging.warning('Found extra stim start. Inferring offset')
    durations = np.array(stimoffs) - np.array(stimstarts)
    
    #find putative behavior session
    behavior_start_ind = np.where((durations>3600)&(durations<3620))[0][0]  
    behavior_start = stimstarts[behavior_start_ind]
    behavior_start_frame = np.where(vf>behavior_start)[0][0]
    logging.warning('Inferred behavior stim start: {}'.format(behavior_start))
    
    #find putative mapping session
    mapping_start_ind = np.where((durations>1500)&(durations<1550))[0][0]  
    mapping_start = stimstarts[mapping_start_ind]
    mapping_start_frame = np.where(vf>mapping_start)[0][0]
    logging.warning('Inferred mapping stim start: {}'.format(mapping_start))

    #find putative replay session
    replay_start_ind = np.where((durations>3600)&(durations<3620))[0][-1]  
    replay_start = stimstarts[replay_start_ind]
    replay_start_frame = np.where(vf>replay_start)[0][0]
    logging.warning('Inferred replay stim start: {}'.format(replay_start))
else:
    behavior_start_frame = 0
    mapping_start_frame = behavior_frame_count
    replay_start_frame = behavior_frame_count + mapping_frame_count

MONITOR_LAG = 0.036
FRAME_APPEAR_TIMES = vf + MONITOR_LAG  

analysis.plot_frame_intervals(vf, behavior_frame_count, mapping_frame_count, 
                              behavior_start_frame, mapping_start_frame,
                              replay_start_frame, FIG_SAVE_DIR) 


### BUILD UNIT TABLE ####
probe_dict = probeSync.build_unit_table(paths['data_probes'], paths, syncDataset)

get_RFs(probe_dict, mapping_data, mapping_start_frame, FRAME_APPEAR_TIMES, FIG_SAVE_DIR)

analysis.plot_population_change_response(probe_dict, behavior_frame_count, mapping_frame_count, trials, 
                                         FRAME_APPEAR_TIMES, FIG_SAVE_DIR, ctx_units_percentile=66)


analysis.plot_running_wheel(behavior_data, mapping_data, replay_data, FIG_SAVE_DIR)