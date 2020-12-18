# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:20:34 2020

@author: svc_ccg
"""
import os, glob
import json
import re

def get_sessions(root, mouseID=None, start_date=None, end_date=None, rig=None, day1=True):
    '''Gets ephys sessions from root directory. 
    Takes only the directories in root with the expected format:
        10 digit lims ID, 6 digit mouseID and 8 digit date
        all separated by underscores: e.g. '1028864423_366122_20200608'
        
    Applies up to four optional filters:
        mouseID: take only sessions from this mouse,
                if mouseID starts with '!' then exclude this mouse
        start_date: take all dates on or after this;
                    date format needs to be 'YYYYMMDD'
        end_date: take all dates before or on this
        rig: which rig experiment was run on ('NP0', 'NP1' most likely)
    ''' 
    if isinstance(root, list):
        in_dir = concatenate_lists([list_dir(r) for r in root])
    else:
        in_dir = list_dir(root)
        
    dirs = [d for d in in_dir if (os.path.isdir(d) \
                                  and validate_session_dir(d))]
    
    for func, criterion in zip([mouseID_filter, start_date_filter, 
                                end_date_filter, rig_filter, day1_filter],
                               [mouseID, start_date, end_date, rig, day1]):
        dirs = apply_filter(dirs, func, criterion)

    return dirs
    

def list_dir(root):
    
    in_dir = os.listdir(root)
    full_paths = [os.path.join(root, d) for d in in_dir]
    return full_paths


def validate_session_dir(d):
    
    base = os.path.basename(d)
    match = re.search('[0-9]{10}_[0-9]{6}_[0-9]{8}', base)
    
    return match is not None


def mouseID_filter(d, mouseID):
    
    base = os.path.basename(d)
    d_mouseID = re.search('_[0-9]{6}_', base).group(0)
    d_mouseID = d_mouseID[1:-1]

    if mouseID[0]=='!':
        return str(d_mouseID) != str(mouseID[1:])
    else:    
        return str(d_mouseID) == str(mouseID)
    
    
def start_date_filter(d, start_date):
    
    base = os.path.basename(d)
    d_date = re.search('_[0-9]{8}', base).group(0)[1:]
    
    return int(d_date)>=int(start_date)
    

def end_date_filter(d, end_date):
    
    base = os.path.basename(d)
    d_date = re.search('_[0-9]{8}', base).group(0)[1:]
    
    return int(d_date)<=int(end_date)


def rig_filter(d, rig):
    
    #find platform json
    platform_file = glob_file(d, '*platform*.json')
    if platform_file is None:
        print('No platform json in {}'.format(d))
        return False
    
    platform_contents = read_json(platform_file)
    d_rig = platform_contents['rig_id']
    
    return all([c in d_rig.upper() for c in rig.upper()])


def day1_filter(d, day1):
    
    filter_out = True
    if day1:
        base = os.path.basename(d)
        lims_id = re.search('[0-9]{10}', base).group(0)
        filter_out = lims_id[0] != '2'
        
    return filter_out


def apply_filter(dirs, filter_func, criterion):
    
    if criterion is None:
        return dirs
    
    else:
        dirs = [d for d in dirs if filter_func(d, criterion)]
        return dirs


def concatenate_lists(lists):
    
    cat = []
    for l in lists:
        if len(l)==0:
            continue
        elif len(l)==1:
            cat.append(l)
        else:
            cat.extend(l)
    return cat

    
def read_json(file_path):
    
    with open(file_path, 'r') as f:
        contents = json.load(f)
    
    return contents


def glob_file(root, format_str):
    
    f = glob.glob(os.path.join(root, format_str))
    if len(f)>0:
        return f[0]
    else:
        print('Could not find file of format'\
              '{} in {}'.format(format_str, root))
        return None
