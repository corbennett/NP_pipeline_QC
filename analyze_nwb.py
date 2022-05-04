# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 23:46:31 2022

@author: svc_ccg
"""
import os, glob
from pynwb import NWBHDF5IO
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import (
    BehaviorEcephysSession)
from numba import njit
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

@njit     
def makePSTH_numba(spikes, startTimes, windowDur, binSize=0.001, convolution_kernel=0.05, avg=True):
    spikes = spikes.flatten()
    startTimes = startTimes - convolution_kernel/2
    windowDur = windowDur + convolution_kernel
    bins = np.arange(0,windowDur+binSize,binSize)
    convkernel = np.ones(int(convolution_kernel/binSize))
    counts = np.zeros(bins.size-1)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/startTimes.size
    counts = np.convolve(counts, convkernel)/(binSize*convkernel.size)
    return counts[convkernel.size-1:-convkernel.size], bins[:-convkernel.size-1]

nwb_base = r'\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\nwbs_220429'
nwb_paths = glob.glob(os.path.join(nwb_base, '*nwb'))

data_dict = {
        'area': [],
        'session': [],
        'change_response': [],
        'image_set': [],
        'shared_change_response':[]}

for inwb, nwb_path in enumerate(nwb_paths):
    with NWBHDF5IO(nwb_path, 'r', load_namespaces=True) as nwb_io:
        session = BehaviorEcephysSession.from_nwb(nwbfile=nwb_io.read())
    
    print('processing file {} of {}'.format(inwb, len(nwb_paths)))
    
    channels = session.get_channels()
    
    units = session.get_units()
    good_unit_filter = ((units['snr']>1)&(units['isi_violations']<1)&(units['firing_rate']>0.1))
    units = units.loc[good_unit_filter]
    unitchannels = units.merge(channels, left_on='peak_channel_id', right_index=True)
    
    spike_times = session.spike_times
    stimulus_presentations = session.stimulus_presentations
    
    change_times = stimulus_presentations.loc[stimulus_presentations['active']&
                                              stimulus_presentations['is_change']&
                                              (~np.isin(stimulus_presentations['image_name'], ['im111_r', 'im083_r']))]
    
    change_times = change_times['start_time'].values
    
    shared_change_times = stimulus_presentations.loc[stimulus_presentations['active']&
                                              stimulus_presentations['is_change']&
                                              (np.isin(stimulus_presentations['image_name'], ['im111_r', 'im083_r']))]
    
    shared_change_times = shared_change_times['start_time'].values
    
    for iu, unit in unitchannels.iterrows():
        
        sts = spike_times[iu]
        area = unit.manual_structure_acronym
        cr = makePSTH_numba(sts, change_times-1, 2, binSize=0.001, convolution_kernel=0.01)
        scr = makePSTH_numba(sts, shared_change_times-1, 2, binSize=0.001, convolution_kernel=0.01)
        
        data_dict['area'].append(area)
        data_dict['session'].append(session.metadata['ecephys_session_id'])
        data_dict['change_response'].append(cr[0])
        data_dict['shared_change_response'].append(scr[0])
        data_dict['image_set'].append(session.task_parameters['session_type'])



H_NOT_NOVEL = [
        1099598937,
        1099869737,
        1104052767,
        1104297538,
        1109680280,
        1109889304,
        1108335514,
        1108528422,
        1116941914,
        1117148442,
        1118324999,
        1118512505,
        1119946360,
        1120251466,
        1130113579,
        1130349290,
        1132595046,
        1132752601,
        1127072792,
        1127256514,
        1131502356,
        1131648544,
        1128520325,
        1128719842,
        1139846596,
        1140102579,
        ]

fig_save_dir = r"C:\Users\svc_ccg\Desktop\Presentations\SAC 2022"
areas_of_interest = ['VISp', 'VISl', 'VISal', 'VISrl', 
                     'VISam', 'VISpm', 'LP', 'LGd', 'TH',
                     'MRN', 'CA3']

data_df = pd.DataFrame(data_dict)

for area in areas_of_interest:
    fig, ax = plt.subplots()
    fig.suptitle(area)
    
    novel = []
    familiar = []
    ainds = np.where(np.array(data_dict['area'])==area)[0]
    print(area, len(ainds))
    for ai in ainds:
        image_set = data_dict['image_set'][ai]
        s_id = data_dict['session'][ai]
        cr = data_dict['change_response'][ai]
        if '_G' in image_set:
            if s_id in H_NOT_NOVEL:
                novel.append(cr)
            else:
                familiar.append(cr)
        else:
            if s_id in H_NOT_NOVEL:
                familiar.append(cr)
            else:
                novel.append(cr)
    
    for color, exp in zip(['r', 'b'], [novel, familiar]):
        mean = np.mean(exp, axis=0)
        sem = np.std(exp, axis=0)/(len(exp)**0.5)
        
        x = np.linspace(-1, 1, len(cr))
        ax.plot(x, mean, color)
        ax.fill_between(x, mean+sem, mean-sem, color=color, alpha=0.25)
    
    ax.set_xlim([-0.2, 0.5])    
    fig.savefig(os.path.join(fig_save_dir, area+'_change_response.pdf'))
    
    
for area in areas_of_interest:
    fig, ax = plt.subplots()
    fig.suptitle(area)
    
    novel_unshared = []
    novel_shared = []
    fam_unshared = []
    fam_shared = []
    ainds = np.where(np.array(data_dict['area'])==area)[0]
    print(area, len(ainds))
    for ai in ainds:
        image_set = data_dict['image_set'][ai]
        s_id = data_dict['session'][ai]
        cr = data_dict['change_response'][ai]
        scr = data_dict['shared_change_response'][ai]
        if '_G' in image_set:
            if s_id in H_NOT_NOVEL:
                novel_unshared.append(cr)
                novel_shared.append(scr)
            else:
                fam_unshared.append(cr)
                fam_shared.append(scr)
        else:
            if s_id in H_NOT_NOVEL:
                fam_unshared.append(cr)
                fam_shared.append(scr)
            else:
                novel_unshared.append(cr)
                novel_shared.append(scr)
    
    for color, exp in zip(['r', 'b', 'orange', 'teal'], 
                          [novel_unshared, fam_unshared, novel_shared, fam_shared]):
        mean = np.mean(exp, axis=0)
        sem = np.std(exp, axis=0)/(len(exp)**0.5)
        
        x = np.linspace(-1, 1, len(cr))
        ax.plot(x, mean, color)
        ax.fill_between(x, mean+sem, mean-sem, color=color, alpha=0.25)
    
    #ax.set_xlim([-0.2, 0.5])    
    fig.savefig(os.path.join(fig_save_dir, area+'_change_response_shared_vs_unshared.pdf'))

    
    