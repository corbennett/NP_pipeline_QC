# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:33:49 2020

@author: svc_ccg
"""

from psycopg2 import connect, extras
#import numpy as np
import os, glob #, shutil
#from visual_behavior.visualization.extended_trials.daily import make_daily_figure
#from visual_behavior.translator.core import create_extended_dataframe
#from visual_behavior.translator.foraging2 import data_to_change_detection_core
#import visual_behavior
#import pandas as pd
#import probeSync_qc as probeSync


class data_getter():
    ''' parent class for data getter, should be able to 
    1) connect to data source
    2) grab experiment data
    3) grab probe data
    '''
    
    def __init__(self, exp_id=None, base_dir=None):
        
        self.data_dict = {}
        self.connect(exp_id, base_dir)
        self.get_exp_data()
        self.get_probe_data()
        
    
    def connect(self):
        pass
    
    
    def get_exp_data(self):
        pass
    
        
    def get_probe_data(self):
        pass
        
    
    
class lims_data_getter(data_getter):
    
    def connect(self, exp_id, base_dir):
        #set up connection to lims
        self.con = connect(
            dbname='lims2',
            user='limsreader',
            host='limsdb2',
            password='limsro',
            port=5432,
        )
        self.con.set_session(
                    readonly=True, 
                    autocommit=True,
                )
        self.cursor = self.con.cursor(
                    cursor_factory=extras.RealDictCursor,
                )

        self.lims_id = exp_id

    
    def get_exp_data(self):
        ''' Get all the experiment files
            eg sync, pkls, videos etc
        '''
        WKF_QRY = '''
            SELECT es.id AS es_id, 
                es.name AS es,
                es.storage_directory,
                es.workflow_state,
                es.date_of_acquisition,
                es.stimulus_name,
                es.foraging_id as foraging_id,
                sp.external_specimen_name,
                isi.id AS isi_experiment_id,
                e.name AS rig,
                u.login AS operator,
                p.code AS project,
                wkft.name AS wkft, 
                wkf.storage_directory || wkf.filename AS wkf_path,
                bs.storage_directory AS behavior_dir
            FROM ecephys_sessions es
                JOIN specimens sp ON sp.id = es.specimen_id
                LEFT JOIN isi_experiments isi ON isi.id = es.isi_experiment_id
                LEFT JOIN equipment e ON e.id = es.equipment_id
                LEFT JOIN users u ON u.id = es.operator_id
                JOIN projects p ON p.id = es.project_id
                LEFT JOIN well_known_files wkf ON wkf.attachable_id = es.id
                LEFT JOIN well_known_file_types wkft ON wkft.id=wkf.well_known_file_type_id
                JOIN behavior_sessions bs ON bs.foraging_id = es.foraging_id
            WHERE es.id = {} 
            ORDER BY es.id
            '''  
        image_qry = '''
            SELECT *
            FROM ecephys_sessions es
            JOIN images im ON im.id=es.id
            WHERE es.id = {}
        '''
        
        self.cursor.execute(WKF_QRY.format(self.lims_id))
        exp_data = self.cursor.fetchall()
        self.data_dict.update(exp_data[0]) #update data_dict to have all the experiment metadata
        [self.data_dict.pop(key) for key in ['wkft', 'wkf_path']] #...but remove the wkf stuff
        
        for e in exp_data:    
            wkft = e['wkft']
            wkf_path = e['wkf_path']
            self.data_dict[wkft] = convert_lims_path(wkf_path)
        
        self.translate_wkf_names()
        
        behavior_dir = convert_lims_path(self.data_dict['behavior_dir'])
        self.data_dict['behavior_pkl'] = glob_file(os.path.join(behavior_dir, '*.pkl'))
        self.data_dict['datestring'] = self.data_dict['date_of_acquisition'].strftime('%Y%m%d')
        self.data_dict['es_id'] = str(self.data_dict['es_id'])
        
        
    def get_probe_data(self):
        ''' Get sorted ephys data for each probe 
        
        TODO: make this actually use the well known file types,
        rather than just grabbing the base directories
        
        '''
        
        WKF_PROBE_QRY = '''
            SELECT es.id AS es_id, 
                es.name AS es, 
                ep.name AS ep, 
                ep.id AS ep_id, 
                wkft.name AS wkft, 
                wkf.storage_directory || wkf.filename AS wkf_path
            FROM ecephys_sessions es
                JOIN ecephys_probes ep ON ep.ecephys_session_id=es.id
                LEFT JOIN well_known_files wkf ON wkf.attachable_id = ep.id
                LEFT JOIN well_known_file_types wkft ON wkft.id=wkf.well_known_file_type_id
            WHERE es.id = {} 
            ORDER BY es.id, ep.name;
            '''
        self.cursor.execute(WKF_PROBE_QRY.format(self.lims_id))
        probe_data = self.cursor.fetchall()
        
        p_info = [p for p in probe_data if p['wkft']=='EcephysSortedProbeInfo']
        probe_bases = [convert_lims_path(os.path.dirname(pi['wkf_path']))[1:] for pi in p_info]
        
        self.data_dict['data_probes'] = []
        for pb in probe_bases:
            probeID = pb[-1]
            self.data_dict['data_probes'].append(probeID)
            self.data_dict['probe' + probeID] = pb
        
        self.probe_data = probe_data
        
        
    def translate_wkf_names(self):
        wkf_dict = {
                'StimulusPickle': 'mapping_pkl',
                'EcephysReplayStimulus': 'replay_pkl',
                'EcephysRigSync': 'sync_file'}
        
        for wkf in wkf_dict:
            if wkf in self.data_dict:
                self.data_dict[wkf_dict[wkf]] = self.data_dict[wkf]
        
        
class local_data_getter(data_getter):
    
    def connect(self, exp_id, base_dir):
        
        if os.path.exists(base_dir):
            self.base_dir = base_dir
        else:
            print('Invalid base directory: ' + base_dir)
        
    
    def get_exp_data(self):
        file_glob_dict = {
                'mapping_pkl': ['*mapping*.pkl', '*stim.pkl'],
                'replay_pkl': '*replay*.pkl',
                'behavior_pkl': '*behavior*.pkl',
                'sync_file': '*.sync',
                'RawEyeTrackingVideo': ['*.eye.avi', '*eye.mp4'],
                'RawBehaviorTrackingVideo': ['*behavior.avi', '*behavior.mp4'],
                'RawFaceTrackingVideo': ['*face.avi', '*face.mp4'],
                'RawEyeTrackingVideoMetadata': '*eye.json',
                'RawBehaviorTrackingVideoMetadata': '*face.json',
                'RawFaceTrackingVideoMetadata': '*behavior.json',
                }
        
        for fn in file_glob_dict:
            if isinstance(file_glob_dict[fn], list):
                paths = [glob_file(os.path.join(self.base_dir, f)) for f in file_glob_dict[fn]]
                path = [p for p in paths if not p is None]
                if len(path)>0:
                    self.data_dict[fn] = path[0]  
            else:
                self.data_dict[fn] = glob_file(os.path.join(self.base_dir, file_glob_dict[fn]))

        
        basename = os.path.basename(self.base_dir)
        self.data_dict['es_id'] = basename.split('_')[0]
        self.data_dict['external_specimen_name'] = basename.split('_')[1]
        self.data_dict['datestring'] = basename.split('_')[2]
        
    def get_probe_data(self):
        self.data_dict['data_probes'] = []
        
        #get probe dirs
        for probeID in 'ABCDEF':
            probe_base = glob_file(os.path.join(self.base_dir, '*probe'+probeID+'_sorted'))
            if probe_base is not None:
                self.data_dict['data_probes'].append(probeID)
                self.data_dict['probe' + probeID] = probe_base
  
        
        

def glob_file(file_path):
    f = glob.glob(file_path)
    if len(f)>0:
        return f[0]
    else:
        return None

def convert_lims_path(path):
    new_path = r'\\' + os.path.normpath(path)[1:]
    return new_path
        
        