# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 09:44:09 2020

@author: svc_ccg
"""
import numpy as np
import os, glob, shutil
import data_getters
import datetime

import logging 



D1_schema = {'es_id': {'minimum_size': None},
 'es': {'minimum_size': None},
 'storage_directory': {'minimum_size': None},
 'workflow_state': {'minimum_size': None},
 'date_of_acquisition': {'minimum_size': None},
 'stimulus_name': {'minimum_size': None},
 'foraging_id': {'minimum_size': None},
 'external_specimen_name': {'minimum_size': None},
 'isi_experiment_id': {'minimum_size': None},
 'rig': {'minimum_size': None},
 'operator': {'minimum_size': None},
 'project': {'minimum_size': None},
 'behavior_dir': {'minimum_size': None},
 'EyeDlcOutputFile': {'minimum_size': 409683458.40000004},
 'EcephysPlatformFile': {'minimum_size': 5947.200000000001},
 'StimulusPickle': {'minimum_size': 7339021.600000001},
 'EcephysRigSync': {'minimum_size': 74143171.2},
 'EcephysSurgeryNotes': {'minimum_size': 1314.4},
 'EcephysReplayStimulus': {'minimum_size': 27547471.200000003},
 'OptoPickle': {'minimum_size': 959931.2000000001},
 'NewstepConfiguration': {'minimum_size': 4641961.600000001},
 'RawFaceTrackingVideo': {'minimum_size': 1955874822.4},
 'RawFaceTrackingVideoMetadata': {'minimum_size': 690.4000000000001},
 'RawEyeTrackingVideo': {'minimum_size': 1956543098.4},
 'RawEyeTrackingVideoMetadata': {'minimum_size': 689.6},
 'RawBehaviorTrackingVideo': {'minimum_size': 1956597323.2},
 'RawBehaviorTrackingVideoMetadata': {'minimum_size': 696.0},
 'EcephysAreaClassifications': {'minimum_size': 611.2},
 'mapping_pkl': {'minimum_size': 7339021.600000001},
 'replay_pkl': {'minimum_size': 27547471.200000003},
 'sync_file': {'minimum_size': 74143171.2},
 'behavior_pkl': {'minimum_size': 18428066.400000002},
 'datestring': {'minimum_size': None},
 'data_probes': {'minimum_size': None},
 'EcephysProbeRawDataABC': {'minimum_size': 1843772.0},
 'EcephysProbeRawDataDEF': {'minimum_size': 1843772.0},
 'EcephysBrainSurfaceLeft': {'minimum_size': 1311537.6},
 'EcephysBrainSurfaceRight': {'minimum_size': 1316493.6},
 'EcephysFiducialImage': {'minimum_size': 1606833.6},
 'EcephysInsertionLocationImage': {'minimum_size': 983416.0},
 'EcephysOverlayImage': {'minimum_size': 542164.0},
 'EcephysPostExperimentLeft': {'minimum_size': 1308762.4000000001},
 'EcephysPostExperimentRight': {'minimum_size': 1321449.6},
 'EcephysPostInsertionLeft': {'minimum_size': 1376120.0},
 'EcephysPostInsertionRight': {'minimum_size': 1391095.2000000002},
 'EcephysPostStimulusLeft': {'minimum_size': 1382569.6},
 'EcephysPostStimulusRight': {'minimum_size': 1393450.4000000001},
 'EcephysPreExperimentLeft': {'minimum_size': 1400624.0},
 'EcephysPreExperimentRight': {'minimum_size': 1421704.0},
 'EcephysPreInsertionLeft': {'minimum_size': 1334457.6},
 'EcephysPreInsertionRight': {'minimum_size': 1339092.8}}



def lims_validation(lims_id):
    
    d = data_getters.lims_data_getter(lims_id)
    paths = d.data_dict
    
    lims_validator = {'D1':{}, 'D2':{}}
    
    for key in D1_schema:
        
        (meets_size_criterion, size, criterion) = validate_schema_entry_size(D1_schema, key, paths)
        
        lims_validator['D1'][key] = {
                'exists': validate_schema_entry_existence(paths, key),
                'file_size': size,
                'min_expected_size': criterion,
                'exceeds_min_expected': meets_size_criterion}
    
    return lims_validator
            

def validate_schema_entry_existence(paths, entry):
    
    if entry not in paths:
        return False
    elif paths[entry] is None:
        return False
    else:
        return True
                    

def validate_schema_entry_size(schema, entry, paths):
    
    min_size = schema[entry]['minimum_size']
    if not min_size:
        return (True, None, None)
    else:
        file_size = get_file_size(paths[entry])
        return (file_size > min_size, file_size, min_size)
    
        
def get_file_size(file):
    
    if file is None:
        return
    
    elif not os.path.exists(file):
        print('File {} does not exist'.format(file))
        return
    
    file_size = os.path.getsize(file)
    return file_size
        
        


#schema = {k: {'minimum_size':None} for k in paths}
#for p in paths:
#    file = paths[p]
#    if not isinstance(file, str):
#        pass
#    elif os.path.isfile(file):
#        size = get_file_size(file)
#        schema[p]['minimum_size'] = 0.8*size
        
        