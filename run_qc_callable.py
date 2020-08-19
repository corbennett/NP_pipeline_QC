# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:42:31 2020

@author: svc_ccg
"""

import numpy as np
import os, glob, shutil
import behavior_analysis
#from visual_behavior.ophys.sync import sync_dataset
from sync_dataset import Dataset as sync_dataset
import pandas as pd
from matplotlib import pyplot as plt
import analysis
import probeSync_qc as probeSync
import scipy.signal
import cv2
import data_getters
from get_RFs_standalone import get_RFs
import logging 


def run_qc(exp_id, save_root):
    
    identifier = exp_id
    if identifier.find('_')>=0:
        d = data_getters.local_data_getter(base_dir=identifier)
    else:
        d = data_getters.lims_data_getter(exp_id=identifier)
    
    paths = d.data_dict
    FIG_SAVE_DIR = os.path.join(save_root, paths['es_id']+'_'+paths['external_specimen_name']+'_'+paths['datestring'])
    if not os.path.exists(FIG_SAVE_DIR):
        os.mkdir(FIG_SAVE_DIR)
    
    figure_prefix = paths['external_specimen_name'] + '_' + paths['datestring'] + '_'
    
    ### GET FILE PATHS TO SYNC AND PKL FILES ###
    SYNC_FILE = paths['sync_file']
    BEHAVIOR_PKL = paths['behavior_pkl']
    REPLAY_PKL = paths['replay_pkl']
    MAPPING_PKL = paths['mapping_pkl']
    
    for f,s in zip([SYNC_FILE, BEHAVIOR_PKL, REPLAY_PKL, MAPPING_PKL], ['sync: ', 'behavior: ', 'replay: ', 'mapping: ']):
        print(s + f)
        
    
    ### GET MAIN DATA STREAMS ###
    syncDataset = sync_dataset(SYNC_FILE)
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
    (behavior_start_frame, mapping_start_frame, replay_start_frame) = probeSync.get_frame_offsets(syncDataset, 
                                                            [behavior_frame_count,
                                                             mapping_frame_count,
                                                             replay_frame_count])
    
    MONITOR_LAG = 0.036 #TO DO: don't hardcode this...
    FRAME_APPEAR_TIMES = vf + MONITOR_LAG  
    
    ### Plot vsync info ###
    vsync_save_dir = os.path.join(FIG_SAVE_DIR, 'vsyncs')
    analysis.plot_frame_intervals(vf, behavior_frame_count, mapping_frame_count, 
                                  behavior_start_frame, mapping_start_frame,
                                  replay_start_frame, vsync_save_dir, prefix=figure_prefix) 
    
    analysis.plot_vsync_interval_histogram(vf, vsync_save_dir)
    analysis.vsync_report(vf, vsync_save_dir)
    
    
    ### BUILD UNIT TABLE ####
    probe_dict = probeSync.build_unit_table(paths['data_probes'], paths, syncDataset)
    
    ### Plot basic unit QC ###
    probe_dirs = [paths['probe'+pid] for pid in paths['data_probes']]
    analysis.plot_unit_quality_hist(probe_dict, os.path.join(FIG_SAVE_DIR, 'unit_quality'), prefix=figure_prefix)
    analysis.plot_unit_distribution_along_probe(probe_dict, os.path.join(FIG_SAVE_DIR, 'unit_quality'), prefix=figure_prefix)
    analysis.plot_all_spike_hist(probe_dirs, FIG_SAVE_DIR, prefix=figure_prefix)
    
    ### Probe/Sync alignment
    analysis.plot_barcode_interval_hist(probe_dirs, syncDataset, os.path.join(FIG_SAVE_DIR, 'unit_quality'), prefix=figure_prefix)
    analysis.probe_sync_report(probe_dirs, syncDataset, os.path.join(FIG_SAVE_DIR, 'unit_quality'), prefix=figure_prefix)
    
    ### Plot visual responses
    get_RFs(probe_dict, mapping_data, mapping_start_frame, FRAME_APPEAR_TIMES, os.path.join(FIG_SAVE_DIR, 'receptive_fields'), prefix=figure_prefix)
    analysis.plot_population_change_response(probe_dict, behavior_frame_count, mapping_frame_count, trials, 
                                             FRAME_APPEAR_TIMES, os.path.join(FIG_SAVE_DIR, 'change_response'), ctx_units_percentile=66, prefix=figure_prefix)
    
    ### Plot running ###
    analysis.plot_running_wheel(behavior_data, mapping_data, replay_data, FIG_SAVE_DIR, prefix=figure_prefix)