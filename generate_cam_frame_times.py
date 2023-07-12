# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 19:05:11 2023

@author: svc_ccg
"""

from probeSync_qc import *
import os, glob
from sync_dataset import Dataset
import pandas as pd
import numpy as np

all_vid_paths = pd.read_csv(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\vbn_video_paths.csv")
all_vbn_exps = pd.read_excel(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\all_np_behavior_mice.xlsx")
save_dir = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\video_frame_times"

exp_ids = []
mouse_ids = []
rig_ids = []
face_times_paths = []
side_times_paths = []

all_truncated_transfers = []
all_excess_frame_times = []
all_transfer_edges = []
all_exposure_edges = []
all_min_tx_rising_interval = []
all_min_tx_falling_interval = []
all_median_tx_rising_interval = []
all_median_tx_falling_interval = []
all_median_exp_rising_interval = []
all_median_exp_falling_interval = []
all_min_exp_rising_interval = []
all_min_exp_falling_interval = []
all_frames_lost = []


def get_sync_intervals(sync_line, sync):
    
    rising = sync.get_rising_edges(sync_line, units='seconds')
    falling = sync.get_falling_edges(sync_line, units='seconds')
    
    falling = falling[falling>rising[0]]
    
    return falling-rising
    
more_truncated_face = all_vid_paths[all_vid_paths['face_truncated_transfers']>all_vid_paths['face_excess_frames']]
all_truncated_inds = []
all_truncated_intervals = []
for ind, row in all_vid_paths.iterrows():
    face_metadata_file = row['face_video_metadata']
    exp_id = os.path.basename(face_metadata_file).split('.')[0]
    exp_info = all_vbn_exps[all_vbn_exps['full_id']==exp_id]   
    mouse_ids.append(exp_info['mouse_id'].values[0])
    rig_ids.append(exp_info['rig'].values[0])
    exp_ids.append(exp_info['full_id'].values[0])
    print(f'Running {exp_id}')
    
    data_dir = os.path.dirname(face_metadata_file)
    sync_file = glob.glob(os.path.join(data_dir, '*.sync'))[0]
    sync = Dataset(sync_file)
    
    for vid, times_paths in zip(['face', 'side'], [face_times_paths, side_times_paths]):
        metadata_file = row[vid + '_video_metadata']
        metadata = read_json(metadata_file)    
    
        frame_times = get_frame_exposure_times(sync, metadata_file)
        total_frames_recorded = metadata['RecordingReport']['FramesRecorded']
        total_frames_lost = metadata['RecordingReport']['FramesLostCount']
        tx_sync_label = 'face_cam_frame_readout' if vid=='face' else 'beh_cam_frame_readout'
        
        tx_intervals = get_sync_intervals(tx_sync_label, sync)
        tx_rising = sync.get_rising_edges(tx_sync_label, units='seconds')
        tx_falling = sync.get_falling_edges(tx_sync_label, units='seconds')

        
        exp_sync_label = 'face_cam_exposing' if vid=='face' else 'beh_cam_exposing'
        exp_rising = sync.get_rising_edges(exp_sync_label, units='seconds')
        exp_falling = sync.get_falling_edges(exp_sync_label, units='seconds')
#        tx_falling = tx_falling[tx_falling>tx_rising[0]]
#        
#        tx_intervals = tx_falling - tx_rising
        tx_mean = np.mean(tx_intervals)
        
        truncated_transfers = np.sum(tx_intervals<tx_mean-0.0001)
        truncated_inds = np.where(tx_intervals<tx_mean - 0.0001)
        truncated_intervals = tx_intervals[tx_intervals<tx_mean-0.0001]
        all_truncated_inds.append(truncated_inds)
        all_truncated_intervals.append(truncated_intervals)
        
        excess_frame_times = len(frame_times) - total_frames_recorded
        all_truncated_transfers.append(truncated_transfers)
        all_excess_frame_times.append(excess_frame_times)
        
        all_transfer_edges.append(tx_rising.size)
        all_exposure_edges.append(exp_rising.size)
        all_min_tx_rising_interval.append(np.min(np.diff(tx_rising)))
        all_min_tx_falling_interval.append(np.min(np.diff(tx_falling)))
        all_min_exp_rising_interval.append(np.min(np.diff(exp_rising)))
        all_min_exp_falling_interval.append(np.min(np.diff(exp_falling)))
        all_median_tx_rising_interval.append(np.median(np.diff(tx_rising)))
        all_median_tx_falling_interval.append(np.median(np.diff(tx_falling)))
        all_median_exp_rising_interval.append(np.median(np.diff(exp_rising)))
        all_median_exp_falling_interval.append(np.median(np.diff(exp_falling)))
        all_frames_lost.append(total_frames_lost)
        
        
        print(f'Found {excess_frame_times} excess frames and {truncated_transfers} truncated transfers')
        
#        frame_times = frame_times[:total_frames_recorded]
#        frame_times = np.insert(frame_times, 0, np.nan) #insert a NaN for the metadata frame
#        np.save(os.path.join(save_dir, exp_id + '_' + vid + '.npy'), frame_times)
#        times_paths.append(os.path.join(save_dir, exp_id + '_' + vid + '.npy'))


all_vid_paths['exp_id'] = exp_ids
all_vid_paths['mouse_id'] = mouse_ids
all_vid_paths['face_timestamp_path'] = face_times_paths
all_vid_paths['side_timestamp_path'] = side_times_paths
all_vid_paths['rig_id'] = rig_ids
all_vid_paths['face_truncated_transfers'] = all_truncated_transfers[::2]
all_vid_paths['side_truncated_transfers'] = all_truncated_transfers[1::2]
all_vid_paths['face_excess_frames'] = all_excess_frame_times[::2]
all_vid_paths['side_excess_frames'] = all_excess_frame_times[1::2]
all_vid_paths['face_min_tx_rising_interval'] = all_min_tx_rising_interval[::2]
all_vid_paths['side_min_tx_rising_interval'] = all_min_tx_rising_interval[1::2]
all_vid_paths['face_min_tx_falling_interval'] = all_min_tx_falling_interval[::2]
all_vid_paths['side_min_tx_falling_interval'] = all_min_tx_falling_interval[1::2]
all_vid_paths['face_min_exp_rising_interval'] = all_min_exp_rising_interval[::2]
all_vid_paths['side_min_exp_rising_interval'] = all_min_exp_rising_interval[1::2]
all_vid_paths['face_min_exp_falling_interval'] = all_min_exp_falling_interval[::2]
all_vid_paths['side_min_exp_falling_interval'] = all_min_exp_falling_interval[1::2]
all_vid_paths['face_median_tx_rising_interval'] = all_median_tx_rising_interval[::2]
all_vid_paths['side_median_tx_rising_interval'] = all_median_tx_rising_interval[1::2]
all_vid_paths['face_median_tx_falling_interval'] = all_median_tx_falling_interval[::2]
all_vid_paths['side_median_tx_falling_interval'] = all_median_tx_falling_interval[1::2]
all_vid_paths['face_median_exp_rising_interval'] = all_median_exp_rising_interval[::2]
all_vid_paths['side_median_exp_rising_interval'] = all_median_exp_rising_interval[1::2]
all_vid_paths['face_median_exp_falling_interval'] = all_median_exp_falling_interval[::2]
all_vid_paths['side_median_exp_falling_interval'] = all_median_exp_falling_interval[1::2]
all_vid_paths['face_frames_lost'] = all_frames_lost[::2]
all_vid_paths['side_frames_lost'] = all_frames_lost[1::2]



all_vid_paths.to_csv(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\vbn_video_paths_full_validation.csv")

