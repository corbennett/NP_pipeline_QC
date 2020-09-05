# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 09:44:09 2020

@author: svc_ccg
"""
import data_getters
from D1_LIMS_schema import D1_schema
from D2_LIMS_schema import D2_schema

import os, sys, json


def run_validation(lims_id, savePath=None):
    
    validator = lims_validation(lims_id)
    if 'Error' in validator:
        D1_report = {'upload_exists':False}
        D2_report = {'upload_exists': False}
    else:
        D1_report = upload_summary(validator['D1'])
        D2_report = upload_summary(validator['D2'])
        
    if savePath:
        master_report = {
                'D1_upload_summary': D1_report,
                'D2_upload_summary': D2_report,
                'file_validation':validator,
                         }
        save_json(master_report, savePath)
    
def lims_validation(lims_id):
    
    try:
        d = data_getters.lims_data_getter(lims_id)
        paths = d.data_dict
        storage_dir = os.path.normpath(paths['storage_directory'])
        lims_validator = {'storage_directory': storage_dir,
                          'D1':{}, 'D2':{}}
        lims_validator['D1'] = check_schema(D1_schema, paths)
        lims_validator['D2'] = check_schema(D2_schema, paths)
    
    except:
        lims_validator = {'Error':str(sys.exc_info()[0]) + 
                          '  ' + str(sys.exc_info()[1]) + 
                          '  ' + str(sys.exc_info()[2])}
    
    return lims_validator


def check_schema(schema, paths):
    
    validation_dict = {}
    for key in schema:
        
        (meets_size_criterion, size, criterion) = validate_schema_entry_size(schema, key, paths)
        
        validation_dict[key] = {
                'exists': validate_schema_entry_existence(paths, key),
                'file_size': size,
                'min_expected_size': criterion,
                'meets_size_criterion': meets_size_criterion}   
    
    return validation_dict


def upload_summary(validator):
    
    report = {'pass': False, 'errors':[]}
    
    exists = []
    meets_size = []
    for entry in validator:
        
        ex = validator[entry]['exists']
        if not ex:
            report['errors'].append('File {} does not exist'.format(entry))
        exists.append(ex)
        
        ms = validator[entry]['meets_size_criterion']
        if ex and not ms:
            report['errors'].append('File {} does not meet size criterion'.format(entry))
        meets_size.append(ms)
    
    report['pass'] = all(exists) & all(meets_size)
    report['upload_exists'] = any(exists)
    
    return report
    

def validate_schema_entry_existence(paths, entry):
    
    if entry not in paths:
        return False
    elif paths[entry] is None:
        return False
#    elif not os.path.exists(paths[entry]):
#        return False
    else:
        return True
                    

def validate_schema_entry_size(schema, entry, paths):
    
    min_size = schema[entry]['minimum_size']
    if not min_size:
        return (True, None, None)
    elif not entry in paths:
        return(False, None, None)
    else:
        file_size = get_file_size(paths[entry])
        return (file_size > min_size, file_size, min_size)
    


def get_file_size(file):
    
    if file is None:
        return
    
    elif not os.path.exists(file):
        print('File {} does not exist'.format(file))
        return -1
    
    file_size = os.path.getsize(file)
    return file_size


def save_json(to_save, save_path):
    
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    with open(save_path, 'w') as f:
        json.dump(to_save, f, indent=2)     
        

        