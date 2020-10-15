# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 09:03:21 2020

@author: svc_ccg
"""

from get_sessions import get_sessions
import subprocess
import re, glob, os
import pickle
import numpy as np
import scipy.ndimage
import scipy.signal
from matplotlib import pyplot as plt
import datetime

source_root = r'\\10.128.50.43\sd6.3'
sources = [r"\\10.128.50.43\sd6.3", r"\\10.128.50.20\sd7"]
sessionsToRun = get_sessions(sources, mouseID='!366122', start_date='20200801')#, end_date='20200922')
destination_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\QC\rf_summary"

#sessionsToRun = get_sessions(source_root, mouseID='!366122', rig='NP1', start_date='20200601')
rf_script = r"C:\Users\svc_ccg\Documents\GitHub\NP_pipeline_QC\get_RFs_standalone.py"
for s in sessionsToRun:
    
    #check whether npy file already exists for this session
    s_base = os.path.basename(s)
    rf_file = glob.glob(os.path.join(destination_folder, s_base+'*'))
    
    if len(rf_file)==0:
        print('running session {}'.format(s))
        command_string = ['python', rf_script, s]
        print(command_string)
        subprocess.call(command_string)
    else:
        print('found existing rf file for session {}'.format(s))
        

data_directory = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\QC\rf_summary'
sessions = os.listdir(data_directory)
sessions = [os.path.join(data_directory,s) for s in sessions if 'npy' in s]

rf_summary = {p:{'peak_chan':[], 'rf_mats':[], 'session':[]} for p in 'ABCDEF'}
for s in sessions:
    
    with open(s, 'rb') as file:
        rf_data = pickle.load(file)
    
    
    for p in rf_data:
        peak_chan_key = 'peakChan' if 'peakChan' in rf_data[p] else 'peak_channel'
        rf_summary[p]['peak_chan'].append(rf_data[p][peak_chan_key])
        rf_summary[p]['rf_mats'].append(rf_data[p]['rfmat'])
        rf_summary[p]['session'].append(s)
        


def get_rf_center_of_mass(rfmat, exp=1):
    
    if rfmat.ndim>2:
        rfmat = np.mean(rfmat, axis=2)
    
    rfmat = rfmat - rfmat.min()
    rfmat = rfmat**exp
    com = scipy.ndimage.center_of_mass(rfmat)
    
    return com[1], com[0]

def get_rf_max_position(rfmat):
    
    if rfmat.ndim>2:
        rfmat = np.mean(rfmat, axis=2)
        
    max_loc = np.unravel_index(np.argmax(rfmat), rfmat.shape)
    
    return max_loc[1], max_loc[0]

def get_significant_rf(rfmat, nreps=1000, conv=2):
    
    if rfmat.ndim>2:
        rfmat = np.mean(rfmat, axis=2)
    
    conv_mat = np.ones((2,2))
    rf_conv = scipy.signal.convolve2d(rfmat, conv_mat, 'same')/4
    
    shuffled = []
    rf_shuff = np.copy(rfmat)
    for rep in np.arange(nreps):
        flat = rf_shuff.flatten()
        np.random.shuffle(flat)
        unflat = flat.reshape([9,9])
        unflat_conv = scipy.signal.convolve2d(unflat, conv_mat, 'same')/4
        shuffled.append(unflat_conv)
    
    shuff_max = [s.max() for s in shuffled]
    percentile_95 = np.percentile(shuff_max, 95)
    
    return rf_conv.max() > percentile_95
    

significant_rfs = {p:[[],[]] for p in 'ABCDEF'}
for p in 'ABCDEF':
    
#    fig, ax = plt.subplots()
#    fig.suptitle(p)
    
    p_rfs = []
    for sess, s in zip(rf_summary[p]['session'], rf_summary[p]['rf_mats']):
        significant = [get_significant_rf(r) for r in s]
        sig_rfs = [r for ir,r in enumerate(s) if significant[ir]]
        p_rfs.extend(sig_rfs)
        significant_rfs[p][0].extend(sig_rfs)
        significant_rfs[p][1].extend([sess]*len(sig_rfs))
    
    
#    #coms = [get_rf_max_position(r) for r in p_rfs]
#    coms =  [get_rf_center_of_mass(r) for r in p_rfs]
#    ax.plot([c[0] for c in coms], [c[1] for c in coms], 'ko', alpha=0.5)
#    ax.set_aspect('equal')
#    ax.set_xlim([0, 8])
#    ax.set_ylim([0, 8])    


for p in 'ABCDEF':
    fig, ax = plt.subplots()
    fig.suptitle(p)
    
    p_rfs = significant_rfs[p][0]
    session_dates = [int(sid.split('_')[-1][:8]) for sid in significant_rfs[p][1]]
    
    split_date = 20201001
    coms =  [get_rf_center_of_mass(r, exp=5) for r, sess in zip(p_rfs, session_dates) if sess<split_date]
    ax.plot([c[0] for c in coms], [c[1] for c in coms], 'ko', alpha=0.3)
    
    coms =  [get_rf_center_of_mass(r, exp=5) for r, sess in zip(p_rfs, session_dates) if sess>=split_date]
    ax.plot([c[0] for c in coms], [c[1] for c in coms], 'ro', alpha=0.7)
    
    ax.set_aspect('equal')
    ax.set_xlim([0, 8])
    ax.set_ylim([0, 8])   
    
    nowstring = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    fig.savefig(os.path.join(data_directory, p + '_all_rfs_' + nowstring + '.png'))


for rf in p_rfs[:20]:
    
    fig, ax = plt.subplots()
    ax.imshow(np.mean(rf, axis=2))
    com = get_rf_center_of_mass(rf, exp=5)
    ax.plot(com[0], com[1], 'ro')    
    
    











    