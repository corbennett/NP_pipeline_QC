# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:31:11 2020

@author: svc_ccg
"""
import numpy as np
from sync_dataset import Dataset as sync_dataset
import visual_behavior
import pandas as pd
import probeSync_qc as probeSync
import data_getters
import build_stim_tables
import h5py

class EcephysBehaviorSession():
    '''Get all data from a Visual Behavior Ecephys experiment.
    Can get data from either LIMS or a local data directory
    '''
    
    @classmethod
    def from_lims(cls, ephys_experiment_id: int):
        file_paths = data_getters.lims_data_getter(exp_id=ephys_experiment_id)
        
        return cls(experiment_info=file_paths.data_dict)

    
    @classmethod
    def from_local(cls, local_path: str, cortical_sort=False):
        file_paths = data_getters.local_data_getter(base_dir=local_path, cortical_sort=cortical_sort)
        
        return cls(experiment_info=file_paths.data_dict)

 
    def __init__(self, experiment_info=None):
        
        self.experiment_info = experiment_info
        
        self.experiment_id = self.experiment_info.get('es_id')
        self.mouse_id = self.experiment_info.get('external_specimen_name')
        self.rig_id = self.experiment_info.get('rig')
        
        self._sync = None
        self._behavior_data = None
        self._mapping_data = None
        self._replay_data = None
        self._opto_data = None
        self._trials = None
        self._unit_table = None
        self._lfp = None
        self._stim_table = None
        self._stim_epochs = None
        self._frame_times = None
        self._lick_times = None
        self._running_speed = None
        self._cam_frame_times = None
        self._opto_stim_table = None
        
    
    @property
    def sync(self):
        
        if self._sync is None:
            sync_file_path = self.experiment_info['sync_file']
            self._sync = sync_dataset(sync_file_path)

        return self._sync
    
    
    @sync.setter
    def sync(self, value):
        self._sync = value
        
    
    @property
    def behavior_data(self):
        
        if self._behavior_data is None:
            self._behavior_data = get_pickle(self.experiment_info['behavior_pkl'])
        
        return self._behavior_data
    
   
    @behavior_data.setter
    def behavior_data(self, value):
        self._behavior_data = value


    @property
    def mapping_data(self):
        
        if self._mapping_data is None:
            self._mapping_data = get_pickle(self.experiment_info['mapping_pkl'])
        
        return self._mapping_data
    
   
    @mapping_data.setter
    def mapping_data(self, value):
        self._mapping_data = value
        
    
    @property
    def replay_data(self):
        
        if self._replay_data is None:
            self._replay_data = get_pickle(self.experiment_info['replay_pkl'])
        
        return self._replay_data
    
   
    @replay_data.setter
    def replay_data(self, value):
        self._replay_data = value
        
    
    @property
    def opto_data(self):
        
        if self._opto_data is None:
            self._opto_data = get_pickle(self.experiment_info['opto_pkl'])
            
        return self._opto_data
    
    
    @opto_data.setter
    def opto_data(self, value):
        self._opto_data = value
        
    
    @property
    def trials(self) -> pd.DataFrame:
        """A dataframe containing behavioral trial start/stop times, and trial
        data of type: pandas.DataFrame"""
        if self._trials is None:
            self._trials = self.api.get_trials()
        return self._trials

    
    @trials.setter
    def trials(self, value):
        self._trials = value
        
    
    @property
    def unit_table(self):

        probes_to_run = self.experiment_info['data_probes']
        if self._unit_table is None:
            self._unit_table = probeSync.build_unit_table(probes_to_run, 
                                                          self.experiment_info, 
                                                          self.sync)

        return self._unit_table
        
    
    @unit_table.setter
    def unit_table(self, value):
        self._unit_table = value
        
        
    @property
    def stim_table(self):
        
        if self._stim_table is None:
            self._stim_table = build_stim_tables.build_full_NP_behavior_stim_table(
                    self.behavior_data, self.mapping_data, 
                    self.replay_data, self.sync)
        
        return self._stim_table


    @stim_table.setter
    def stim_table(self, value):
        self._stim_table = value
        
    
    @property
    def lfp(self):
        
        lfp_dirs = [self.experiment_info['lfp'+pid] for pid in self.experiment_info['data_probes']]
        if self._lfp is None:
            self._lfp = probeSync.build_lfp_dict(lfp_dirs, self.sync)
         
        return self._lfp
            
     
    @lfp.setter
    def lfp(self, value):
        self._lfp = value
        
        
    @property
    def stim_epochs(self):
        
        if self._stim_epochs is None:
            
            behavior_frame_count = self.behavior_data['items']['behavior']['intervalsms'].size + 1
            mapping_frame_count = self.mapping_data['intervalsms'].size + 1
            replay_frame_count = self.replay_data['intervalsms'].size + 1
            
            start_frames = probeSync.get_frame_offsets(self.sync, 
                                                      [behavior_frame_count,
                                                       mapping_frame_count,
                                                       replay_frame_count], 
                                                       tolerance=0)
            end_frames = [start_frames[it]+total-1 for it, total in enumerate([behavior_frame_count,
                                                       mapping_frame_count,
                                                       replay_frame_count])]
            
            self._stim_epochs = {epoch:[start,end] for epoch,start,end in zip(['behavior', 'mapping', 'replay'],
                                                                             start_frames, end_frames)}   
    
        return self._stim_epochs
    

    @stim_epochs.setter
    def stim_epochs(self, value):
        self._stim_epochs = value
        
    
    @property
    def frame_times(self):
        
        if self._frame_times is None:
            self._frame_times = probeSync.get_vsyncs(self.sync, fallback_line=2)
        
        return self._frame_times
    

    @frame_times.setter
    def frame_times(self, value):
        self._frame_times = value
        
    
    @property
    def running_speed(self):
        
        if self._running_speed is None:
            self._running_speed[0] = np.concatenate([get_running_from_pkl(pkl) for pkl in 
                               [self.behavior_data, self.mapping_data, self.replay_data]])
    
            self._running_speed[1] = self.frame_times
            
        return self._running_speed

    
    @running_speed.setter
    def running_speed(self, value):
        self._running_speed = value
    
        
    @property
    def cam_frame_times(self):
        
        eye_cam_dict = {'Eye': 'RawEyeTrackingVideo',
                        'Face': 'RawFaceTrackingVideo', 
                        'Side': 'RawBehaviorTrackingVideo'}
        
        
        if self._cam_frame_times is None:
            self._cam_frame_times = {}
            for cam in eye_cam_dict:
                cam_json = self.experiment_info.get(eye_cam_dict[cam] + 'Metadata')
                if cam_json:
                    cam_frame_times = probeSync.get_frame_exposure_times(self.sync, cam_json)
                    self._cam_frame_times[cam] = cam_frame_times
        return self._cam_frame_times
    

    @cam_frame_times.setter
    def cam_frame_times(self, value):
        self._cam_frame_times = value


    @property
    def lick_times(self):
        
        if self._lick_times is None:
            self._lick_times = probeSync.get_lick_times(self.sync)
        
        return self._lick_times
    

    @lick_times.setter
    def lick_times(self, value):
        self._lick_times = value


    @property
    def reward_times(self):
        
        if self._reward_times is None:
            reward_frames = self.behavior_data['items']['behavior']['rewards'][0]['reward_times'][:, 1]
            self._reward_times = self.frame_times[reward_frames.astype(int)]
            
        return self._reward_times
            
    
    @reward_times.setter
    def reward_times(self, value):
        self._reward_times = value
                   
    
    @property
    def opto_stim_table(self):
        
        if self._opto_stim_table is None:
            self._opto_stim_table = build_stim_tables.get_opto_stim_table(self.sync, self.opto_data)

        return self._opto_stim_table

    @opto_stim_table.setter
    def opto_stim_table(self, value):
        self._opto_stim_table = value
        

def get_running_from_pkl(pkl):
    
    key = 'behavior' if 'behavior' in pkl['items'] else 'foraging'
    intervals = pkl['items']['behavior']['intervalsms'] if 'intervalsms' not in pkl else pkl['intervalsms']
    time = np.insert(np.cumsum(intervals), 0, 0)/1000.
    
    dx,vsig,vin = [pkl['items'][key]['encoders'][0][rkey] for rkey in ('dx','vsig','vin')]
    run_speed = visual_behavior.analyze.compute_running_speed(dx[:len(time)],time,vsig[:len(time)],vin[:len(time)])
    return run_speed


def get_pickle(pickle_path):
    return pd.read_pickle(pickle_path)


def save_to_h5(obj, savepath):
    pass
    
    
    
    
    
    
    
    
    