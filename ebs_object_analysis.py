# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:13:19 2021

@author: svc_ccg
"""

from collections import OrderedDict
import analysis
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

change_times = stim_table.loc[stim_table['change']==1, 'Start'].values

for p in units:
    
    probe_df = units[p].loc[units[p]['quality'] == 'good']
    
    for ir, row in probe_df.iterrows():
        
        if row['firing_rate']>0.1 and row['peakChan']>200:
            times = row['times']
            
            psth = analysis.makePSTH_numba(times, change_times-1, 2)
            fig, ax = plt.subplots()
            fig.suptitle(str(row['peakChan']) + ' ' + str(row['isi_viol']))
            plt.plot(psth[1], psth[0])
            
            
def add_optotagged_to_df(unit_df, opto_stim_table):
    
    all_opto_psths = []
    for irow, row in unit_df.iterrows():
        
        opto_psths = get_opto_response(row['times'], opto_stim_table)
        all_opto_psths.append(opto_psths)
        
    for ind, (irow, row) in enumerate(unit_df.iterrows()):
        
        optotagged = get_optotagged(all_opto_psths[ind])
        unit_df.loc[irow,'optotagged'] = optotagged
        

def get_optotagged(opto_psths):
    
    high_long = opto_psths[-1]
    high_short = opto_psths[int(len(opto_psths)/2 - 1)]
    
    baseline_long = high_long[0][:200]
    above_baseline_long = np.sum(high_long[0][350:550]>(baseline_long.mean()+baseline_long.std()*5))
    
    baseline_short = high_short[0][:100]
    above_baseline_short = np.sum(high_short[0][100:120]>(baseline_short.mean()+baseline_short.std()*5))
    
    optotagged=False
    if above_baseline_long>100 and above_baseline_short>5:
        optotagged=True
    
    return optotagged
    

def get_opto_response(spikes, opto_stim_table):
    
    levels = np.unique(opto_stim_table['trial_levels'])
    conds = np.unique(opto_stim_table['trial_conditions'])
    trial_start_times = opto_stim_table['trial_start_times']
    
    condition_psths = []
    cond_trial_duration = [0.2, 1.2]
    cond_conv_kernel = [0.002, 0.01]
    for ic, cond in enumerate(conds):
        kernel_size = cond_conv_kernel[ic]
        plot_duration = cond_trial_duration[ic]
        
        for il, level in enumerate(levels):
            trial_inds = (opto_stim_table['trial_levels']==level) & (opto_stim_table['trial_conditions']==cond)
            trial_starts = trial_start_times[trial_inds]
            psth = analysis.makePSTH_numba(spikes.flatten(), trial_starts-0.1, plot_duration, 
                                        binSize=0.001, convolution_kernel=0.001, avg=True)
            condition_psths.append(psth)
    
    return condition_psths

    
def plot_opto_responses(unit_df, opto_stim_table, opto_sample_rate=10000, save_opto_mats=False):
    
    levels = np.unique(opto_stim_table['trial_levels'])
    conds = np.unique(opto_stim_table['trial_conditions'])
    
    trial_start_times = opto_stim_table['trial_start_times']
    probes = unit_df['probe'].unique()
    opto_mats_dict = {p:{} for p in probes}
    for probe in probes:
        u_df = unit_df.loc[unit_df['probe']==probe]
        
        good_units = u_df[(u_df['quality']=='good')&(u_df['snr']>1)]
        spikes = good_units['times']
        peakChans = good_units['peak_channel'].values
        unit_shank_order = np.argsort(peakChans)
        opto_mats_dict[probe]['peak_channels'] = peakChans[unit_shank_order]
        
        fig = plt.figure(constrained_layout=True, facecolor='w')
        fig.set_size_inches([18,10])
        fig.suptitle('Probe {} opto responses'.format(probe))
        gs = gridspec.GridSpec(levels.size*2 + 1, conds.size*10+1, figure=fig)
        #gs = gridspec.GridSpec(levels.size*2 + 1, conds.size, figure=fig)
        color_axes = []
        ims = []
        cond_trial_duration = [0.2, 1.2]
        cond_conv_kernel = [0.002, 0.01]
        for ic, cond in enumerate(conds):
            kernel_size = cond_conv_kernel[ic]
            #this_waveform = opto_pkl['opto_waveforms'][cond]
            plot_duration = cond_trial_duration[ic]
            ax_wave = fig.add_subplot(gs[0, ic*10:(ic+1)*10])
            #ax_wave.plot(np.arange(this_waveform.size)/opto_sample_rate, this_waveform)
            ax_wave.set_xlim([-0.1, plot_duration-0.1])
            ax_wave.set_xticks(np.linspace(0, plot_duration-0.1, 3))
            ax_wave.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)
            ax_wave.spines['top'].set_visible(False)
            ax_wave.spines['right'].set_visible(False)
            
            if ic==1:
                ax_wave.set_yticks([])
                ax_wave.spines['left'].set_visible(False)
            
            for il, level in enumerate(levels):
                
                trial_inds = (opto_stim_table['trial_levels']==level) & (opto_stim_table['trial_conditions']==cond)
                trial_starts = trial_start_times[trial_inds]
                psths = np.array([analysis.makePSTH_numba(s.flatten(), trial_starts-0.1, plot_duration, 
                                        binSize=0.001, convolution_kernel=kernel_size, avg=True) for s in spikes])
        
                #bin_times = psths[0, 1, :]
                psths = psths[unit_shank_order, 0, :].squeeze()
                psths_baseline_sub = np.array([p-np.mean(p[:100]) for p in psths])   
                opto_mats_dict[probe][str(cond) + '_' + str(level)] = psths_baseline_sub
                ax = fig.add_subplot(gs[2*il+1:2*il+3, ic*10:(ic+1)*10])
                im = ax.imshow(psths_baseline_sub, origin='lower', interpolation='none', aspect='auto')
                ax.set_title('Level: {}'.format(level))
                color_axes.append(ax)
                ims.append(im)
                #plt.colorbar(im)
                if il==len(levels)-1:
                    ax.set_xticks(np.linspace(100, 1000*plot_duration, 3))
                    ax.set_xticklabels(np.linspace(0, 1000*plot_duration-100, 3))
                    ax.set_xlabel('Time from LED onset (ms)')
                    if ic==0:
                        ax.set_ylabel('Unit # sorted by depth')
                    
                else:
                    ax.set_xticks([])
                
                if ic==1:
                    ax.set_yticks([])
        
#        min_clim_val = np.min([im.get_clim()[0] for im in ims])
#        max_clim_val = np.max([im.get_clim()[1] for im in ims])
        
        min_clim_val = -5
        max_clim_val = 50
        
        for im in ims:
            im.set_clim([min_clim_val, max_clim_val])    
            
        xs, ys = np.meshgrid(np.arange(2), np.arange(min_clim_val, max_clim_val))
        ax_colorbar = fig.add_subplot(gs[-2:, conds.size*10:])
        ax_colorbar.imshow(ys, origin='lower', clim=[min_clim_val, max_clim_val])
        ax_colorbar.set_yticks([0, np.round(max_clim_val - min_clim_val)])
        ax_colorbar.set_yticklabels(np.round([min_clim_val, max_clim_val], 2))
        
        #ax_colorbar.set_aspect(2)
        ax_colorbar.set_ylabel('spikes relative to baseline')
        #ax_colorbar.yaxis.set_label_position('right')
        ax_colorbar.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        ax_colorbar.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=True,
            labelright=True,
            labelleft=False)
        
#        save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix+probe+'_optoResponse.png'))
#        #save_as_plotly_json(fig, os.path.join(FIG_SAVE_DIR, prefix+probe+'_optoResponse.plotly.json'))
#        if save_opto_mats:
#            for probe in opto_mats_dict:
#                np.savez(os.path.join(FIG_SAVE_DIR, prefix+probe+'_optomat.npz'), **opto_mats_dict[probe])
#
#


### look at change triggered running speed ###
running = ee.running_speed['running_speed']
change_times = ee.stim_table.loc[ee.stim_table['change']==1, 'Start'].values
frame_times = ee.frame_times['frame_times']

ctr = []
for ct in change_times:
    
    rind = np.searchsorted(frame_times, ct)
    r = running[0][rind-120:rind+120]
    ctr.append(r)

plt.plot(np.mean(ctr, axis=0))


#### compute change responses for each image
image_list = stim_table['image_name'].dropna().unique()
def imagewise_change_response(spikes, image_list, stim_table, monitor_lag=0.02):
    
    active_changes = stim_table.loc[(stim_table['change']==1)&(stim_table['active'])]
    
    im_changes = []
    for im in image_list:
        if im == 'omitted':
            trigger_times = stim_table.loc[(stim_table['omitted'])&(stim_table['active']), 'Start'].values + 0.016 #seems to be a 1 frame problem here...
        else:  
            trigger_times = active_changes.loc[active_changes['image_name']==im, 'Start'].values
            
        psth, _ = analysis.makePSTH_numba(spikes.flatten(), trigger_times-1, 2)
        im_changes.append(psth)
            
    return im_changes
    
ot = ee2.unit_table.loc[ee2.unit_table['optotagged']]
o_imc = ot.apply(lambda row: imagewise_change_response(row['times'], image_list, ee2.stim_table), axis=1)
fig, ax = plt.subplots()
for ind, im in enumerate(image_list):
    
    all_cells = [o[ind] for o in o_imc]
    ax.plot(np.mean(all_cells, axis=0))

ax.legend(image_list)

def get_ctx_inds(unit_table):
    probe_grouped = unit_table.groupby('probe')
    ctx_inds = []
    for probe, pgroup in probe_grouped:
        
        top_channel = pgroup['peak_channel'].max()
        bottom_ctx = top_channel - 70
        pctx = pgroup.loc[pgroup['peak_channel']>bottom_ctx]
        ctx_inds.extend(pctx.index.values)
    return ctx_inds

ctx = ee2.unit_table.loc[ctx_inds]
#ctx = ctx.loc[ctx['optotagged']==0]

all_imc = ctx.apply(lambda row: imagewise_change_response(row['times'], image_list, ee2.stim_table), axis=1)
fig, ax = plt.subplots()
for ind, im in enumerate(image_list):
    
    all_cells = [o[ind] for o in ac]
    ax.plot(np.mean(all_cells, axis=0))

ax.legend(image_list)




import EcephysBehaviorSession as ebs
import os
import pandas as pd

h5_dir = r"C:\Data\NP_pipeline_h5s"
h5_list = [os.path.join(h5_dir, h) for h in os.listdir(h5_dir)]

failed_h5 = []
for ih, h5 in enumerate(h5_list):
    
    try:
        print('loading: {}'.format(h5))
        ee2 = ebs.EcephysBehaviorSession.from_h5(h5)
        image_list = np.sort(ee2.stim_table['image_name'].dropna().unique())
        ctx_inds = get_ctx_inds(ee2.unit_table)
        ctx_df = ee2.unit_table.loc[ctx_inds]
        
        change_responses = ctx_df.apply(lambda row: imagewise_change_response(row['times'], image_list, ee2.stim_table, ee2.experiment_info['monitor_lag']), axis=1)
        ctx_df['change_responses'] = change_responses
        
        opto_responses = ctx_df.apply(lambda row: get_opto_response(row['times'], ee2.opto_stim_table), axis=1)
        ctx_df['opto_responses'] = opto_responses
        
        ctx_df['optotagged'] = ctx_df.apply(lambda row: get_optotagged(row['opto_responses']), axis=1)
        ctx_df['image_set'] = ee2.stim_table.loc[ee2.stim_table['active'], 'stimulus_name'].iloc[0]
        ctx_df['genotype'] = ee2.experiment_info['genotype']
        ctx_df['mouseID'] = ee2.experiment_info['external_specimen_name']
        ctx_df['sessionID'] = ee2.experiment_info['es_id']
        ctx_df['date'] = ee2.experiment_info['datestring']
        ctx_df['image_list'] = [image_list]*len(ctx_df)
        
        abbrev_ut = ctx_df.drop(columns=['times', 'template'])
        
        if ih==0:
            combined_df = abbrev_ut.copy()
        else:
            combined_df = pd.concat([combined_df, abbrev_ut])
    except Exception as e:
        print('failed to add {} due to {}'.format(h5, e))
        failed_h5.append(h5)

combined_df['unit_session_id'] = combined_df.index.astype(str) + combined_df['sessionID']

import h5py
savepath = r"C:\Data\NP_pipeline_h5s\popdata3.h5"
with h5py.File(savepath,'a') as savefile:
    
    grp = savefile['/']
    ebs.add_to_hdf5(savefile, grp=grp, saveDict=combined_df.set_index('unit_session_id').to_dict())
    
    
