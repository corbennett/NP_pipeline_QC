# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:08:45 2020

@author: svc_ccg
"""

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


class run_qc():
    
    def __init__(self, exp_id, save_root, modules_to_run='all'):
        
        self.modules_to_run = modules_to_run
        
        identifier = exp_id
        if identifier.find('_')>=0:
            d = data_getters.local_data_getter(base_dir=identifier)
        else:
            d = data_getters.lims_data_getter(exp_id=identifier)
        
        self.paths = d.data_dict
        self.FIG_SAVE_DIR = os.path.join(save_root, self.paths['es_id']+'_'+ self.paths['external_specimen_name']+'_'+ self.paths['datestring'])
        if not os.path.exists(self.FIG_SAVE_DIR):
            os.mkdir(self.FIG_SAVE_DIR)
        
        self.figure_prefix = self.paths['external_specimen_name'] + '_' + self.paths['datestring'] + '_'

        ### GET FILE PATHS TO SYNC AND PKL FILES ###
        self.SYNC_FILE = self.paths['sync_file']
        self.BEHAVIOR_PKL = self.paths['behavior_pkl']
        self.REPLAY_PKL = self.paths['replay_pkl']
        self.MAPPING_PKL = self.paths['mapping_pkl']

        for f,s in zip([self.SYNC_FILE, self.BEHAVIOR_PKL, self.REPLAY_PKL, self.MAPPING_PKL], ['sync: ', 'behavior: ', 'replay: ', 'mapping: ']):
            print(s + f)

        ### GET MAIN DATA STREAMS ###
        self.syncDataset = sync_dataset(self.SYNC_FILE)
        self.behavior_data = pd.read_pickle(self.BEHAVIOR_PKL)
        self.mapping_data = pd.read_pickle(self.MAPPING_PKL)
        self.replay_data = pd.read_pickle(self.REPLAY_PKL)
        
        self.trials = behavior_analysis.get_trials_df(self.behavior_data)
    
        ### CHECK FRAME COUNTS ###
        vr, self.vf = probeSync.get_sync_line_data(self.syncDataset, channel=2)
    
        self.behavior_frame_count = self.behavior_data['items']['behavior']['intervalsms'].size + 1
        self.mapping_frame_count = self.mapping_data['intervalsms'].size + 1
        self.replay_frame_count = self.replay_data['intervalsms'].size + 1
        
        self.total_pkl_frames = (self.behavior_frame_count +
                            self.mapping_frame_count +
                            self.replay_frame_count) 
        
        ### CHECK THAT NO FRAMES WERE DROPPED FROM SYNC ###
        print('frames in pkl files: {}'.format(self.total_pkl_frames))
        print('frames in sync file: {}'.format(len(self.vf)))
        
        #assert(total_pkl_frames==len(vf))
        
        ### CHECK THAT REPLAY AND BEHAVIOR HAVE SAME FRAME COUNT ###
        print('frames in behavior stim: {}'.format(self.behavior_frame_count))
        print('frames in replay stim: {}'.format(self.replay_frame_count))
        
        #assert(behavior_frame_count==replay_frame_count)
        
        # look for potential frame offsets from aborted stims
        (self.behavior_start_frame, self.mapping_start_frame, self.replay_start_frame) = probeSync.get_frame_offsets(
                                                                self.syncDataset, 
                                                                [self.behavior_frame_count,
                                                                 self.mapping_frame_count,
                                                                 self.replay_frame_count])
        
        self.behavior_end_frame = self.behavior_start_frame + self.behavior_frame_count - 1
        self.mapping_end_frame = self.mapping_start_frame + self.mapping_frame_count - 1
        self.replay_end_frame = self.replay_start_frame + self.replay_frame_count - 1
        
        MONITOR_LAG = 0.036 #TO DO: don't hardcode this...
        self.FRAME_APPEAR_TIMES = self.vf + MONITOR_LAG  
        
        self.behavior_start_time, self.mapping_start_time, self.replay_start_time = [self.FRAME_APPEAR_TIMES[f] for f in 
                                                                      [self.behavior_start_frame, self.mapping_start_frame, self.replay_start_frame]]
        self.behavior_end_time, self.mapping_end_time, self.replay_end_time = [self.FRAME_APPEAR_TIMES[f] for f in 
                                                                      [self.behavior_end_frame, self.mapping_end_frame, self.replay_end_frame]]
        self.probe_dirs = [self.paths['probe'+pid] for pid in self.paths['data_probes']]
        self.probe_dict = None
        self.lfp_dict = None
        
        self._run_modules()
        
    
    def _run_modules(self):
        
        self.errors = []
        module_list = [func for func in dir(self) if callable(getattr(self, func))]
        for module in module_list:
            if module[0] == '_':
                continue
            
            if module in self.modules_to_run or self.modules_to_run=='all':
                func = getattr(self, module)
                print('\n' + '#'*20)
                print('Running module: {}\n'.format(module))
                try:
                    func()
                except Exception as e:
                    print('Error running module {}'.format(module))
                    self.errors.append((module, e))
            
          
    def _build_unit_table(self):
        ### BUILD UNIT TABLE ####
        self.probe_dict = probeSync.build_unit_table(self.paths['data_probes'], self.paths, self.syncDataset)
    
    
    def _build_lfp_dict(self):
        self.lfp_dict = probeSync.build_lfp_dict(self.probe_dirs, self.syncDataset)

    
    def behavior(self):
        ### Behavior Analysis ###
        behavior_plot_dir = os.path.join(self.FIG_SAVE_DIR, 'behavior')
        behavior_analysis.plot_behavior(self.trials, behavior_plot_dir, prefix=self.figure_prefix)
        
        trial_types, counts = behavior_analysis.get_trial_counts(self.trials)
        behavior_analysis.plot_trial_type_pie(counts, trial_types, behavior_plot_dir, prefix=self.figure_prefix)
        
        
    def vsync(self):
        ### Plot vsync info ###
        vsync_save_dir = os.path.join(self.FIG_SAVE_DIR, 'vsyncs')
        analysis.plot_frame_intervals(self.vf, self.behavior_frame_count, self.mapping_frame_count, 
                                      self.behavior_start_frame, self.mapping_start_frame,
                                      self.replay_start_frame, vsync_save_dir, prefix=self.figure_prefix) 
        analysis.plot_vsync_interval_histogram(self.vf, vsync_save_dir, prefix = self.figure_prefix)
        analysis.vsync_report(self.vf, self.total_pkl_frames, vsync_save_dir, prefix = self.figure_prefix)
        
    
    def probe_yield(self):
        ### Plot Probe Yield QC ###
        if self.probe_dict is None:
            self._build_unit_table()
        
        probe_yield_dir = os.path.join(self.FIG_SAVE_DIR, 'probe_yield')
        analysis.plot_unit_quality_hist(self.probe_dict, probe_yield_dir, prefix=self.figure_prefix)
        analysis.plot_unit_distribution_along_probe(self.probe_dict, probe_yield_dir, prefix=self.figure_prefix)
        analysis.plot_all_spike_hist(self.probe_dict, probe_yield_dir, prefix=self.figure_prefix+'good')
        analysis.copy_probe_depth_images(self.paths, probe_yield_dir, prefix=self.figure_prefix)
    
    
    def unit_metrics(self):
        ### Unit Metrics ###
        unit_metrics_dir = os.path.join(self.FIG_SAVE_DIR, 'unit_metrics')
        analysis.plot_unit_metrics(self.paths, unit_metrics_dir, prefix=self.figure_prefix)
        
    
    def probe_sync_alignment(self):
        ### Probe/Sync alignment
        probeSyncDir = os.path.join(self.FIG_SAVE_DIR, 'probeSyncAlignment')
        analysis.plot_barcode_interval_hist(self.probe_dirs, self.syncDataset, probeSyncDir, prefix=self.figure_prefix)
        analysis.plot_barcode_intervals(self.probe_dirs, self.syncDataset, probeSyncDir, prefix=self.figure_prefix)
        analysis.probe_sync_report(self.probe_dirs, self.syncDataset, probeSyncDir, prefix=self.figure_prefix)
        analysis.plot_barcode_matches(self.probe_dirs, self.syncDataset, probeSyncDir, prefix=self.figure_prefix)
    
    
    def receptive_fields(self):
        ### Plot receptive fields
        if self.probe_dict is None:
            self._build_unit_table()
        get_RFs(self.probe_dict, self.mapping_data, self.mapping_start_frame, self.FRAME_APPEAR_TIMES, 
                os.path.join(self.FIG_SAVE_DIR, 'receptive_fields'), prefix=self.figure_prefix)
    
    
    def change_responses(self):
        if self.probe_dict is None:
            self._build_unit_table()
        analysis.plot_population_change_response(self.probe_dict, self.behavior_start_frame, self.replay_start_frame, self.trials, 
                                             self.FRAME_APPEAR_TIMES, os.path.join(self.FIG_SAVE_DIR, 'change_response'), ctx_units_percentile=66, prefix=self.figure_prefix)
    
    def running_wheel(self):
        ### Plot running ###
        analysis.plot_running_wheel(self.behavior_data, self.mapping_data, self.replay_data, 
                                    self.behavior_plot_dir, prefix=self.figure_prefix)
    
    
    def lfp(self, agarChRange=None, num_licks=20, windowBefore=0.5, 
            windowAfter=1.5, min_inter_lick_time = 0.5, behavior_duration=3600):
        
        ### LFP ###
        if self.lfp_dict is None:
            self._build_lfp_dict()
        lfp_save_dir = os.path.join(self.FIG_SAVE_DIR, 'LFP')
        lick_times = analysis.get_rewarded_lick_times(probeSync.get_lick_times(self.syncDataset), 
                                                      self.FRAME_APPEAR_TIMES[self.behavior_start_frame:], self.trials, min_inter_lick_time=min_inter_lick_time)
        analysis.plot_lick_triggered_LFP(self.lfp_dict, lick_times, lfp_save_dir, prefix=self.figure_prefix, 
                                agarChRange=agarChRange, num_licks=num_licks, windowBefore=windowBefore, 
                                windowAfter=windowAfter, min_inter_lick_time = min_inter_lick_time, behavior_duration=behavior_duration)
    
    def probe_targeting(self):
        
        targeting_dir = os.path.join(self.FIG_SAVE_DIR, 'probe_targeting')
        images_to_copy = ['insertion_location_image', 'overlay_image']
        analysis.copy_files(images_to_copy, self.paths, targeting_dir)
                
            
    def videos(self, frames_for_each_epoch=[2,2,2]):
        ### VIDEOS ###
        video_dir = os.path.join(self.FIG_SAVE_DIR, 'videos')
        analysis.lost_camera_frame_report(self.paths, video_dir, prefix=self.figure_prefix)
        analysis.camera_frame_grabs(self.paths, self.syncDataset, video_dir, 
                                    [self.behavior_start_time, self.mapping_start_time, self.replay_start_time],
                                    [self.behavior_end_time, self.mapping_end_time, self.replay_end_time],
                                     epoch_frame_nums = frames_for_each_epoch, prefix=self.figure_prefix)

    