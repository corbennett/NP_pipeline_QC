# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 17:25:46 2020

@author: svc_ccg
"""
import numpy as np
import os
#from visual_behavior.ophys.sync import sync_dataset
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import analysis
import probeSync_qc as probeSync
import data_getters
import logging
import sys
# sys.path.append("..")
from sync_dataset import Dataset as sync_dataset


def get_RFs(probe_dict, mapping_data, first_frame_offset, FRAME_APPEAR_TIMES, FIG_SAVE_DIR, ctx_units_percentile = 40, return_rfs=False, response_thresh=20): 
    
    ### PLOT POPULATION RF FOR EACH PROBE ###
    rfs = {p:{k:[] for k in ['peakChan', 'unitID', 'rfmat']} for p in probe_dict}
    for p in probe_dict:
        try:
            print(f'########## Getting RFs for probe {p} ###########')
            u_df = probe_dict[p]
            good_units = u_df[(u_df['quality']=='good')&(u_df['snr']>1)]
         
            ctx_bottom_chan = np.percentile(good_units['peak_channel'], 100-ctx_units_percentile)
            spikes = good_units.loc[good_units['peak_channel']>ctx_bottom_chan]
            rmats = []
            for ind, s in spikes.iterrows():
                rmat = analysis.plot_rf(mapping_data, s['times'].flatten(), first_frame_offset, FRAME_APPEAR_TIMES)
                if rmat.max()>response_thresh:
                    rfs[p]['peakChan'].append(s['peakChan'])
                    rfs[p]['unitID'].append(s['Unnamed: 0'])
                    rfs[p]['rfmat'].append(rmat)
                    rmats.append(rmat/rmat.max())
                
            rmats_normed_mean = np.nanmean(rmats, axis=0)
         
            
            fig = plt.figure(constrained_layout=True, figsize=[6,6])
            title = p + ' population RF'
            fig.suptitle(title, color='w')
            
            nrows, ncols = 10,10
            gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
            
            ax1 = fig.add_subplot(gs[0:nrows-1, 0:ncols-1])
            ax2 = fig.add_subplot(gs[0:nrows-1, ncols-1])
            ax3 = fig.add_subplot(gs[nrows-1, 0:ncols-1])
            
            ax1.imshow(np.mean(rmats_normed_mean, axis=2), origin='lower')
            ax1.set_xticks([], minor=[])
            ax1.set_yticks([], minor=[])
            
            ax3.imshow(np.vstack((np.arange(-45, 46), np.arange(-45, 46))), cmap='jet', clim=[-60, 60])
            ax3.set_xticks([0, 45, 90])
            ax3.set_xticklabels([-45, 0, 45])
            ax3.set_yticks([], minor=[])
            ax3.set_xlabel('Azimuth')
            
            ax2.imshow(np.hstack((np.arange(-45, 46)[:,None], np.arange(-45, 46)[:,None])), cmap='jet_r', clim=[-60, 60])
            ax2.yaxis.tick_right()
            ax2.set_yticks([0, 45, 90])
            ax2.set_yticklabels([-45, 0, 45])
            ax2.set_xticks([], minor=[])
            ax2.yaxis.set_label_position("right")
            ax2.set_ylabel('Elevation', rotation=270)
            
            fig.savefig(os.path.join(FIG_SAVE_DIR, title + '.png'))
            
        
        except Exception as E:
            logging.error(f'{p} failed: {E}')
            print(E)
    
    if return_rfs:
        return rfs
                    

if __name__ == "__main__":
    
    # run as standalone script
    experiment_id = sys.argv[1]
    print(experiment_id)
    
    d = data_getters.local_data_getter(base_dir=experiment_id)
    paths = d.data_dict
    
    FIG_SAVE_DIR = os.path.join(r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\QC", 
                                paths['es_id']+'_'+paths['external_specimen_name']+'_'+paths['datestring'])
    
    if not os.path.exists(FIG_SAVE_DIR):
        os.mkdir(FIG_SAVE_DIR)
    
    ### GET FILE PATHS TO SYNC AND PKL FILES ###
    SYNC_FILE = paths['sync_file']
    # BEHAVIOR_PKL = paths['behavior_pkl']
    # REPLAY_PKL = paths['replay_pkl']
    MAPPING_PKL = paths['mapping_pkl']

    try:
        syncDataset = sync_dataset(SYNC_FILE)
    except Exception as e:
        logging.error('Error reading sync file: {}'.format(e))

    try:
        mapping_data = pd.read_pickle(MAPPING_PKL)
    except Exception as e:
        logging.error('Error reading mapping pkl file: {}'.format(e))

    # replay_data = pd.read_pickle(REPLAY_PKL)
    
    
    ### PLOT FRAME INTERVALS ###
    vr, vf = probeSync.get_sync_line_data(syncDataset, channel=2)
    
    # behavior_frame_count = behavior_data['items']['behavior']['intervalsms'].size + 1
    mapping_frame_count = mapping_data['intervalsms'].size + 1
    # replay_frame_count = replay_data['intervalsms'].size + 1
    
    MONITOR_LAG = 0.036
    FRAME_APPEAR_TIMES = vf + MONITOR_LAG  
    
    ### CHECK THAT NO FRAMES WERE DROPPED FROM SYNC ###
    # total_pkl_frames = (behavior_frame_count +
    #                     mapping_frame_count +
    #                     replay_frame_count) 

    # print('frames in pkl files: {}'.format(total_pkl_frames))
    # print('frames in sync file: {}'.format(len(vf)))
    
    #infer start frames for stimuli
    start_frame = probeSync.get_frame_offsets(syncDataset, [mapping_frame_count])
    
    if start_frame is not None:
        print('RF mapping started at frame {}, or experiment time {} seconds'.format(start_frame[0], start_frame[0]/60.))
        
        probe_dict = probeSync.build_unit_table(paths['data_probes'], paths, syncDataset)
        get_RFs(probe_dict, mapping_data, start_frame[0], FRAME_APPEAR_TIMES, FIG_SAVE_DIR)

    else:
        logging.error('Could not find mapping stim start frame')


    
