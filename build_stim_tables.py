# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:50:27 2020

@author: svc_ccg
"""

import pandas as pd
from allensdk.brain_observatory.behavior.stimulus_processing import get_stimulus_presentations
from allensdk.brain_observatory.sync_dataset import Dataset
from allensdk.brain_observatory.ecephys.stimulus_table.ephys_pre_spikes import build_stimuluswise_table
from allensdk.brain_observatory.ecephys.file_io.stim_file import (
    CamStimOnePickleStimFile,
)
import numpy as np
import logging

def get_frame_offsets(sync_dataset, behavior_frame_count, mapping_frame_count):
    # look for potential frame offsets from aborted stims
    vf = get_vsyncs(sync_dataset)
    stimstarts, stimoffs = get_stim_starts_ends(sync_dataset)
    
    if len(stimstarts)>3:
        logging.warning('Found extra stim start. Inferring offset')
        durations = np.array(stimoffs) - np.array(stimstarts)
        
        #find putative behavior session (first stim with ~3600 frames)
        behavior_start_ind = np.where((durations>3600)&(durations<3620))[0][0]  
        behavior_start = stimstarts[behavior_start_ind]
        behavior_start_frame = np.where(vf>behavior_start)[0][0]
        logging.warning('Inferred behavior stim start: {}'.format(behavior_start))
        
        #find putative mapping session (first stim with ~1500 frames)
        mapping_start_ind = np.where((durations>1500)&(durations<1550))[0][0]  
        mapping_start = stimstarts[mapping_start_ind]
        mapping_start_frame = np.where(vf>mapping_start)[0][0]
        logging.warning('Inferred mapping stim start: {}'.format(mapping_start))
    
        #find putative replay session (last stim with ~3600 frames)
        replay_start_ind = np.where((durations>3600)&(durations<3620))[0][-1]  
        replay_start = stimstarts[replay_start_ind]
        replay_start_frame = np.where(vf>replay_start)[0][0]
        logging.warning('Inferred replay stim start: {}'.format(replay_start))
    else:
        behavior_start_frame = 0
        mapping_start_frame = behavior_frame_count
        replay_start_frame = behavior_frame_count + mapping_frame_count
    
    return behavior_start_frame, mapping_start_frame, replay_start_frame



def generate_behavior_stim_table(pkl_path, sync_path, frame_offset=0):
    
    p = pd.read_pickle(pkl_path)
    image_set =  p['items']['behavior']['params']['stimulus']['params']['image_set']
    image_set = image_set.split('/')[-1].split('.')[0]
    num_frames = p['items']['behavior']['intervalsms'].size + 1
    
    sync_dataset = Dataset(sync_path)
    frame_timestamps = get_vsyncs(sync_dataset)
    
    stim_table = get_stimulus_presentations(p, frame_timestamps[frame_offset:frame_offset+num_frames])
    stim_table['stimulus_block'] = 0
    stim_table['stimulus_name'] = image_set
    stim_table = stim_table.rename(columns={'frame':'start_frame', 'start_time':'Start', 'stop_time':'End'})
        
    change = np.zeros(len(stim_table))
    repeat_number = np.zeros(len(stim_table))
    current_image = stim_table.iloc[0]['stimulus_name']
    for index, row in stim_table.iterrows():
        if row['image_name'] != current_image:
            change[index] = 1
            repeat_number[index] = 0
            current_image = row['image_name']
        else:
            if row['omitted']:
                repeat_number[index] = repeat_number[index-1]
            else:
                repeat_number[index] = repeat_number[index-1] + 1
            
    stim_table['change'] = change.astype(int)
    stim_table.loc[0, 'change'] = 0
    stim_table['flashes_since_change'] = repeat_number.astype(int)
    
    stim_table[['change', 'flashes_since_change', 'image_name', 'omitted']].head(60)
    
    return stim_table

    
def generate_replay_stim_table(pkl_path, sync_path, behavior_stim_table, stimulus_block=2, frame_offset=0):
    
    p = pd.read_pickle(pkl_path)
    num_frames = p['intervalsms'].size + 1
    
    sync_dataset = Dataset(sync_path)
    frame_timestamps = get_vsyncs(sync_dataset)
    frame_timestamps = frame_timestamps[frame_offset:frame_offset+num_frames]
    
    ims = p['stimuli'][0]['sweep_params']['ReplaceImage'][0]
    im_names = np.unique([img for img in ims if img is not None])
    
    ## CHECK THAT REPLAY MATCHES BEHAVIOR
    im_ons = []
    im_offs = []
    im_names = []
    for ind, im in enumerate(ims):
        if ind==0:
            continue
        elif ind<len(ims)-1:
            if ims[ind-1] is None and ims[ind] is not None:
                im_ons.append(ind)
                im_names.append(im)
            elif ims[ind] is not None and ims[ind+1] is None:
                im_offs.append(ind)
     
    inter_flash_interval = np.diff(im_ons)
    putative_omitted = np.where(inter_flash_interval>70)[0]
    im_names_with_omitted = np.insert(im_names, putative_omitted+1, 'omitted')
    
    assert all(behavior_stim_table['image_name'] == im_names_with_omitted)
    
    ## IF SO, JUST USE THE BEHAVIOR STIM TABLE, BUT ADJUST TIMES/FRAMES
    stim_table = behavior_stim_table.copy(deep=True)
    stim_table['stimulus_block'] = stimulus_block
    stim_table['Start'] = frame_timestamps[stim_table['start_frame']]
    stim_table.loc[stim_table['omitted']==False, 'End'] = frame_timestamps[stim_table['end_frame'].dropna().astype(int)]
    stim_table['start_frame'] = stim_table['start_frame'] + frame_offset
    stim_table.loc[stim_table['omitted']==False, 'end_frame'] = stim_table['end_frame'] + frame_offset
    
    return stim_table

def generate_mapping_stim_table(pkl_path, sync_path, stimulus_block=1, frame_offset=0):
    
    stim_file = CamStimOnePickleStimFile.factory(pkl_path)
    
    seconds_to_frames = (
        lambda seconds: (np.array(seconds) + stim_file.pre_blank_sec)
        * stim_file.frames_per_second
    )
    
    stim_table = build_stimuluswise_table(stim_file.stimuli[0], seconds_to_frames)[0]
    
    
def get_vsyncs(sync_dataset, fallback_line=2):
    
    lines = sync_dataset.line_labels
    
    #look for vsyncs in labels
    vsync_line = fallback_line
    for line in lines:
        if 'vsync' in line:
            vsync_line = line
    
    falling_edges = sync_dataset.get_falling_edges(vsync_line, units='seconds')
    
    return falling_edges

        
def get_stim_starts_ends(sync_dataset, fallback_line=5):
    
    lines = sync_dataset.line_labels
    
    #look for vsyncs in labels
    if 'stim_running' in lines:
        stim_line = 'stim_running'
    else:
        stim_line = fallback_line
    
    stim_ons = sync_dataset.get_rising_edges(stim_line, units='seconds')
    stim_offs = sync_dataset.get_falling_edges(stim_line, units='seconds')
    
    return stim_ons, stim_offs
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        