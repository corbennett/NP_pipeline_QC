# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:31:11 2020

@author: svc_ccg
"""
import numpy as np
import os, glob, shutil
import json
import behavior_analysis
#from visual_behavior.ophys.sync import sync_dataset
from sync_dataset import Dataset as sync_dataset
import pandas as pd
import analysis
import probeSync_qc as probeSync
import data_getters
from get_RFs_standalone import get_RFs
import logging 
from query_lims import query_lims


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
    def trials(self) -> pd.DataFrame:
        """A dataframe containing behavioral trial start/stop times, and trial
        data
        :rtype: pandas.DataFrame"""
        if self._trials is None:
            self._trials = self.api.get_trials()
        return self._trials

    @trials.setter
    def trials(self, value):
        self._trials = value
        

def get_pickle(pickle_path):
    return pd.read_pickle(pickle_path)


