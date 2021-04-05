# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:13:19 2021

@author: svc_ccg
"""

from collections import OrderedDict
import analysis
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import scipy.stats
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

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


def get_mean_change_response(df, image_list):
    
    crs = df.get('change_responses')
    mean_cr = []
    for ind, im in enumerate(image_list):
        
        all_cells = [c[ind] for c in crs]
        mean_cr.append(np.mean(all_cells, axis=0))
    
    return mean_cr
    
def get_all_change_responses(df, image_list, orderbydepth=False):
    
    if orderbydepth:
        crs = df.sort_values(by=['peak_channel'], ascending=False).get('change_responses')
    else:
        crs = df.get('change_responses')
    all_cr = []
    for ind, im in enumerate(image_list):
        
        all_cells = [c[ind] for c in crs]
        all_cr.append(np.array(all_cells))
    
    return np.array(all_cr)

  

import EcephysBehaviorSession as ebs
import os
import pandas as pd

h5_dir = r"C:\Data\NP_pipeline_h5s"
h5_list = [os.path.join(h5_dir, h) for h in os.listdir(h5_dir)]

failed_h5 = []
for ih, h5 in enumerate(h5_list):
    
    try:
        print('loading: {}  {}/{}'.format(h5, ih, len(h5_list)))
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
        
        abbrev_ut = ctx_df.drop(columns=['amplitudes', 'template'])
        
        stim_table = ee2.stim_table
        stim_table['sessionID'] = ee2.experiment_info['es_id']
        stim_table['mouseID'] = ee2.experiment_info['external_specimen_name']
        
        
        if ih==0:
            combined_df = abbrev_ut.copy()
            combined_stim_df = stim_table.copy()
        else:
            combined_df = pd.concat([combined_df, abbrev_ut])
            combined_stim_df = pd.concat([combined_stim_df, stim_table])
    
    except Exception as e:
        print('failed to add {} due to {}'.format(h5, e))
        failed_h5.append(h5)

combined_df['unit_session_id'] = combined_df.index.astype(str) + combined_df['sessionID']
combined_stim_df['stim_session_index'] = combined_stim_df.index.astype(str) + '_' + combined_stim_df['sessionID']

### save combined DFs ###
import h5py
savepath = r"C:\Data\NP_pipeline_h5s\popdata_with_times.h5"
with h5py.File(savepath,'a') as savefile:
    grp = savefile['/']
    ebs.add_to_hdf5(savefile, grp=grp, saveDict=combined_df.set_index('unit_session_id').to_dict())

savepath = r"C:\Data\NP_pipeline_h5s\popdata_stim_tables.h5"
with h5py.File(savepath,'a') as savefile:
    grp = savefile['/']
    ebs.add_to_hdf5(savefile, grp=grp, saveDict=combined_stim_df.set_index('stim_session_index').to_dict())
  

### LOAD POPDATA FROM MEMORY ####
loadfrommem = False
if loadfrommem:
    combined_df = {}
    ebs.hdf5_to_dict(combined_df, r"C:\Data\NP_pipeline_h5s\popdata_with_times.h5")

if loadfrommem:
    combined_stim_df = {}
    ebs.hdf5_to_dict(combined_stim_df, r"C:\Data\NP_pipeline_h5s\popdata_stim_tables.h5")


### ANALYSIS ###
g_image_list = combined_df.loc[combined_df['image_set'].str.contains('_G'),'image_list'].iloc[0]
h_image_list = combined_df.loc[combined_df['image_set'].str.contains('_H'),'image_list'].iloc[0]

g_cr = get_mean_change_response(combined_df.loc[combined_df['image_set'].str.contains('_G')], g_image_list)
h_cr = get_mean_change_response(combined_df.loc[combined_df['image_set'].str.contains('_H')], h_image_list)

fig, ax = plt.subplots()
for c in g_cr:
    ax.plot(c)
ax.legend(g_image_list)

fig, ax = plt.subplots()
for c in h_cr:
    ax.plot(c)
ax.legend(h_image_list)

def formataxes(ax, title=None, xLabel=None, yLabel=None, 
               xTickLabels=None, yTickLabels=None, no_spines=False,
               ylims=None, xlims=None, spinesToHide=None):
    
    if spinesToHide is None:
        spinesToHide = ['right', 'top', 'left', 'bottom'] if no_spines else ['right', 'top']
    for spines in spinesToHide:
        ax.spines[spines].set_visible(False)

    ax.tick_params(direction='out',top=False,right=False)
    
    if title is not None:
        ax.set_title(title)
    if xLabel is not None:
        ax.set_xlabel(xLabel)
    if yLabel is not None:
        ax.set_ylabel(yLabel)
    if ylims is not None:
        ax.set_ylim(ylims)
    if xlims is not None:
        ax.set_xlim(xlims)
        

## mean change response across all probes split by genotype/image set
cr_dict = {}
for genotype in combined_df['genotype'].unique():
    cr_dict[genotype] = {}
    for ind, (image_set, image_list) in enumerate(zip(['_G', '_H'], [g_image_list, h_image_list])):
        
        if genotype in ['Vip-IRES-Cre;Ai32', 'Sst-IRES-Cre;Ai32']:
            df_filter = ((combined_df['image_set'].str.contains(image_set)) &
                        (combined_df['genotype']== genotype) &
                        (combined_df['optotagged']))
            
        else:
            df_filter = ((combined_df['image_set'].str.contains(image_set)) &
                         (combined_df['optotagged']==False))
        
        
        cr = get_mean_change_response(combined_df.loc[df_filter], image_list)
        cr_dict[genotype][image_set] = cr



for genotype in cr_dict:
    fig, axes = plt.subplots(1,2)
    for ind, (image_set, image_list) in enumerate(zip(['_G', '_H'], [g_image_list, h_image_list])):
        
        cr = cr_dict[genotype][image_set]
        for ic, c in enumerate(cr):
            if image_list[ic] in ['im083_r', 'im111_r']:
                color = 'r'
            elif image_list[ic] == 'omitted':
                color = 'c'
            else:
                color = '0.5'
            axes[ind].plot(c, color, alpha=0.5)

        
        axes[ind].legend(image_list)
        axes[ind].set_title(genotype + ' ' + image_set)

### Mean change response for genotype/probe/image set combos
genotypes = ['Vip-IRES-Cre;Ai32', 'Sst-IRES-Cre;Ai32', 'RS', 'FS']
probe_grouped = combined_df.groupby('probe')
p_cr_dict = {}
for probe, pgroup in probe_grouped:
    
    p_cr_dict[probe] = {}
    for genotype in genotypes:
        p_cr_dict[probe][genotype] = {}
        for ind, (image_set, image_list) in enumerate(zip(['_G', '_H'], [g_image_list, h_image_list])):
            
            good_unit_filter = ((pgroup['snr']>1)&(pgroup['isi_viol']<1)&(pgroup['firing_rate']>0.1))
            if genotype in ['Vip-IRES-Cre;Ai32', 'Sst-IRES-Cre;Ai32']:
                df_filter = ((pgroup['image_set'].str.contains(image_set)) &
                            (pgroup['genotype']== genotype) &
                            (pgroup['optotagged']))
                
            elif genotype=='RS':
                df_filter = ((pgroup['image_set'].str.contains(image_set)) &
                             (pgroup['optotagged']==False) &
                             (pgroup['duration']>=0.4))
            elif genotype=='FS':
                df_filter = ((pgroup['image_set'].str.contains(image_set)) &
                             (pgroup['optotagged']==False) &
                             (pgroup['duration']<0.4))
            
            cr = get_mean_change_response(pgroup.loc[df_filter&good_unit_filter], image_list)
            p_cr_dict[probe][genotype][image_set]= {'cr':cr, 'unit_count':len(pgroup.loc[df_filter&good_unit_filter])}


fig_save_path = r"C:\Users\svc_ccg\Desktop\Presentations\Tuesday Seminar 03232021"
    
for probe in p_cr_dict:
    for genotype in genotypes:
        fig, axes = plt.subplots(1,2)
        fig.set_size_inches([12,8])
        for ind, (image_set, image_list) in enumerate(zip(['_G', '_H'], [g_image_list, h_image_list])):
            
            cr = p_cr_dict[probe][genotype][image_set]['cr']
            for ic, c in enumerate(cr):
                if image_list[ic] in ['im083_r', 'im111_r']:
                    color = 'r'
                elif image_list[ic] == 'omitted':
                    color = 'c'
                else:
                    color = '0.5'
                axes[ind].plot(c, color, alpha=0.5)
    
            
            #axes[ind].legend(image_list)
            axes[ind].set_title(genotype + ' ' + image_set + ' ' + probe + ': ' + str(p_cr_dict[probe][genotype][image_set]['unit_count']) + ' units')
        
        miny = np.min([a.get_ylim()[0] for a in axes])
        maxy = np.max([a.get_ylim()[1] for a in axes])
        
        [a.set_ylim([miny, maxy]) for a in axes]
        
        fig.savefig(os.path.join(fig_save_path, 'mean_image_response_' + probe + '_' + genotype + '_goodunits.png'))



### Image response mat for all cells of each genotype/probe/image set combo
fig_save_path = r"C:\Users\svc_ccg\Desktop\Presentations\Tuesday Seminar 03232021"

genotypes = ['Vip-IRES-Cre;Ai32', 'Sst-IRES-Cre;Ai32', 'RS', 'FS']
probe_grouped = combined_df.groupby('probe')
p_crmat_dict = {}
for probe, pgroup in probe_grouped:
    
    p_crmat_dict[probe] = {}
    for genotype in genotypes:
        p_crmat_dict[probe][genotype] = {}
        for ind, (image_set, image_list) in enumerate(zip(['_G', '_H'], [g_image_list, h_image_list])):
            
            good_unit_filter = ((pgroup['snr']>1)&(pgroup['isi_viol']<1)&(pgroup['firing_rate']>0.1))
            gtoh_filter = (pgroup['mouseID']!='548722')
            if genotype in ['Vip-IRES-Cre;Ai32', 'Sst-IRES-Cre;Ai32']:
                df_filter = ((pgroup['image_set'].str.contains(image_set)) &
                            (pgroup['genotype']== genotype) &
                            (pgroup['optotagged']))
                
            elif genotype=='RS':
                df_filter = ((pgroup['image_set'].str.contains(image_set)) &
                             (pgroup['optotagged']==False) &
                             (pgroup['duration']>=0.4))
            elif genotype=='FS':
                df_filter = ((pgroup['image_set'].str.contains(image_set)) &
                             (pgroup['optotagged']==False) &
                             (pgroup['duration']<0.4))
            
            crs = get_all_change_responses(pgroup.loc[df_filter&good_unit_filter&gtoh_filter], image_list, orderbydepth=True)
            p_crmat_dict[probe][genotype][image_set] = crs

common_ind_dict = {'_G': [5,6],
                  '_H': [3,6]}
other_ind_dict = {'_G': [0,1,2,3,4,7],
                  '_H': [0,1,2,4,5,7]}
genotypes = ['Vip-IRES-Cre;Ai32', 'Sst-IRES-Cre;Ai32', 'RS', 'FS']
for probe in ['C']:# p_crmat_dict:
    for genotype in genotypes:
        for ind, (image_set, image_list) in enumerate(zip(['_G', '_H'], [g_image_list, h_image_list])):
            for cat, cat_inds in zip(['common', 'other'], [common_ind_dict, other_ind_dict]):
                cr = np.mean(p_crmat_dict[probe][genotype][image_set][cat_inds[image_set]], axis=0)
                if len(cr)>0:
                    cr_sorted = cr[np.argsort(np.max(cr[:, 1050:1150], axis=1))]
                    
                    fig, ax = plt.subplots()
                    fig.suptitle(probe+' ' +genotype+' '+ image_set+ ' ' +cat)
                    ax.imshow(cr_sorted[:,1000:1400], clim=[0,50], origin='lower')
                    figname = probe+'_'+genotype+'_'+'cr_responsemat'+image_set+'.pdf'
                    analysis.save_figure(fig, os.path.join(fig_save_path, 'cr_cell_mats\\'+figname))
            plt.close('all')

## plot CR responses to common and private images group by both sharedness and image set
genotypes = ['Vip-IRES-Cre;Ai32', 'Sst-IRES-Cre;Ai32', 'RS', 'FS']
image_set_colors = ['b', 'r']
shared_colors = ['purple', 'g']
for probe in p_crmat_dict:
    for genotype in genotypes:
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches([12,8])
        
        ifig, iax = plt.subplots(1,2)
        ifig.set_size_inches([12,8])
        for ind, (image_set, image_list) in enumerate(zip(['_G', '_H'], [g_image_list, h_image_list])):
            iax[ind].set_title(probe+' '+genotype+image_set)
            for ic, (cat, cat_inds) in enumerate(zip(['common', 'private'], [common_ind_dict, other_ind_dict])):
                cr = np.mean(p_crmat_dict[probe][genotype][image_set][cat_inds[image_set]], axis=0)
                ax[ic].set_title(probe+' '+genotype+' '+cat)
                
                if len(cr)>0:
                    time = np.arange(-1000,1000)
                    cr = cr - np.mean(cr[:, 900:1000], axis=1)[:, None]
                    cr_mean_over_cells = np.mean(cr, axis=0)
                    cr_sem_over_cells = np.std(cr, axis=0)/(cr.shape[0])**0.5
                    ax[ic].plot(time, cr_mean_over_cells, image_set_colors[ind])
                    ax[ic].fill_between(time, cr_mean_over_cells+cr_sem_over_cells,
                      cr_mean_over_cells-cr_sem_over_cells, color=image_set_colors[ind], alpha=0.5, linewidth=0)
                    formataxes(ax[ic], xLabel='Time from change (s)', yLabel='Firing Rate (Hz)')
                    
                    iax[ind].plot(time, cr_mean_over_cells, shared_colors[ic])
                    iax[ind].fill_between(time, cr_mean_over_cells+cr_sem_over_cells,
                      cr_mean_over_cells-cr_sem_over_cells, color=shared_colors[ic], alpha=0.5, linewidth=0)
                    formataxes(iax[ind], xLabel='Time from change (s)', yLabel='Firing Rate (Hz)')
                    
                    
        miny = np.min([a.get_ylim()[0] for a in ax])
        maxy = np.max([a.get_ylim()[1] for a in ax])
        
        [[a.set_ylim([miny, maxy]) for a in axes] for axes in [ax, iax]]
        [[a.set_xlim([-100, 500]) for a in axes] for axes in [ax, iax]]
        ax[0].legend(['G', 'H'])
        iax[0].legend(['common', 'private'])
        figname = probe + '_' + genotype + '_commonVsprivate_groupedbyshared.pdf'
        analysis.save_figure(fig, os.path.join(fig_save_path, 'commonVprivate_CR_groupedbyshared_newcolors\\'+figname))
        ifigname = probe + '_' + genotype + '_commonVsprivate_groupedbyimageset.pdf'
        analysis.save_figure(ifig, os.path.join(fig_save_path, 'commonVprivate_CR_groupedbyimageset_newcolors\\'+figname))
            #plt.close(fig)
        plt.close('all')

## COLLAPSE ACROSS PROBES plot CR responses to common and private images group by both sharedness and image set
genotypes = ['Vip-IRES-Cre;Ai32', 'Sst-IRES-Cre;Ai32', 'RS', 'FS']
image_set_colors = ['b', 'r']
shared_colors = ['purple', 'g']
for genotype in genotypes:
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches([12,8])
    
    ifig, iax = plt.subplots(1,2)
    ifig.set_size_inches([12,8])
    for ind, (image_set, image_list) in enumerate(zip(['_G', '_H'], [g_image_list, h_image_list])):
        iax[ind].set_title(genotype+image_set)
        for ic, (cat, cat_inds) in enumerate(zip(['common', 'private'], [common_ind_dict, other_ind_dict])):
            crs = [p_crmat_dict[probe][genotype][image_set][cat_inds[image_set]] for probe in 'ABCDEF']
            crs = [cr for cr in crs if cr.size>0]
            crs = np.concatenate(crs, axis=1)
            cr = np.mean(crs, axis=0)
            ax[ic].set_title(genotype+' '+cat)
            
            if len(cr)>0:
                time = np.arange(-1000,1000)
                cr = cr - np.mean(cr[:, 900:1000], axis=1)[:, None]
                cr_mean_over_cells = np.mean(cr, axis=0)
                cr_sem_over_cells = np.std(cr, axis=0)/(cr.shape[0])**0.5
                ax[ic].plot(time, cr_mean_over_cells, image_set_colors[ind])
                ax[ic].fill_between(time, cr_mean_over_cells+cr_sem_over_cells,
                  cr_mean_over_cells-cr_sem_over_cells, color=image_set_colors[ind], alpha=0.5)
                formataxes(ax[ic], xLabel='Time from change (s)', yLabel='Firing Rate (Hz)')
                
                iax[ind].plot(time, cr_mean_over_cells, shared_colors[ic])
                iax[ind].fill_between(time, cr_mean_over_cells+cr_sem_over_cells,
                  cr_mean_over_cells-cr_sem_over_cells, color=shared_colors[ic], alpha=0.5)
                formataxes(iax[ind], xLabel='Time from change (s)', yLabel='Firing Rate (Hz)')
                
                
    miny = np.min([a.get_ylim()[0] for a in ax])
    maxy = np.max([a.get_ylim()[1] for a in ax])
    
    [[a.set_ylim([miny, maxy]) for a in axes] for axes in [ax, iax]]
    [[a.set_xlim([-100, 500]) for a in axes] for axes in [ax, iax]]
    ax[0].legend(['G', 'H'])
    iax[0].legend(['common', 'private'])
    figname = 'allprobes_' + genotype + '_commonVsprivate_groupedbyshared.png'
    analysis.save_figure(fig, os.path.join(fig_save_path, 'commonVprivate_CR_groupedbyshared\\'+figname))
    ifigname = 'allprobes_' + genotype + '_commonVsprivate_groupedbyimageset.png'
    analysis.save_figure(ifig, os.path.join(fig_save_path, 'commonVprivate_CR_groupedbyimageset\\'+figname))
        #plt.close(fig)
    plt.close('all')

#### PLOT RESPONSE TO OMISSION
genotypes = ['Vip-IRES-Cre;Ai32', 'Sst-IRES-Cre;Ai32', 'RS', 'FS']
image_set_colors = ['b', 'r']
shared_colors = ['purple', 'g']
for probe in p_crmat_dict:
    for genotype in genotypes:
        fig, ax = plt.subplots()
        fig.set_size_inches([6,6])
        
        for ind, (image_set, image_list) in enumerate(zip(['_G', '_H'], [g_image_list, h_image_list])):
            iax[ind].set_title(probe+' '+genotype+image_set)
            
            omission_resp = p_crmat_dict[probe][genotype][image_set][-1]
            ax.set_title(probe+' '+genotype+' '+cat)
            
            if len(omission_resp)>0:
                time = np.arange(-1000,1000)
                or_mean_over_cells = np.mean(omission_resp, axis=0)
                or_sem_over_cells = np.std(omission_resp, axis=0)/(omission_resp.shape[0])**0.5
                ax.plot(time, or_mean_over_cells, image_set_colors[ind])
                ax.fill_between(time, or_mean_over_cells+or_sem_over_cells,
                  or_mean_over_cells-or_sem_over_cells, color=image_set_colors[ind], alpha=0.5, linewidth=0.0)
                formataxes(ax, xLabel='Time from omission (s)', yLabel='Firing Rate (Hz)')
                
        ax.set_xlim([-400, 1000]) 
        ax.legend(['G', 'H'])
        
        figname = probe + '_' + genotype + '_omission_response.pdf'
        analysis.save_figure(fig, os.path.join(fig_save_path, 'omission_response_newcolors\\'+figname))
        plt.close('all')



### OPTO responses for all genotype/probe combos
fig_save_path = r"C:\Users\svc_ccg\Desktop\Presentations\Tuesday Seminar 03232021"

genotypes = ['Vip-IRES-Cre;Ai32', 'Sst-IRES-Cre;Ai32']
probe_grouped = combined_df.groupby('probe')
for probe, pgroup in probe_grouped:
    
    #opto_mat_dict[probe] = {}
    for genotype in genotypes:
        for ind, (image_set, image_list) in enumerate(zip(['_G', '_H'], [g_image_list, h_image_list])):
            
            good_unit_filter = ((pgroup['snr']>1)&(pgroup['isi_viol']<1)&(pgroup['firing_rate']>0.1))
            
            if genotype in ['Vip-IRES-Cre;Ai32', 'Sst-IRES-Cre;Ai32']:
                df_filter = ((pgroup['image_set'].str.contains(image_set)) &
                            (pgroup['genotype']== genotype) &
                            (pgroup['optotagged']))
                
            elif genotype=='RS':
                df_filter = ((pgroup['image_set'].str.contains(image_set)) &
                             (pgroup['optotagged']==False) &
                             (pgroup['duration']>=0.4))
            elif genotype=='FS':
                df_filter = ((pgroup['image_set'].str.contains(image_set)) &
                             (pgroup['optotagged']==False) &
                             (pgroup['duration']<0.4))
            
            ors = pgroup.loc[df_filter&good_unit_filter].get('opto_responses')
            
            or_pulse = [o[int(len(o)/2)-1][0] for o in ors] 
            or_sin = [o[-1][0] for o in ors]
            
            if len(or_pulse)>0:
                fig, ax = plt.subplots(1,2)
                fig.suptitle(probe+' '+genotype+' '+image_set)
                ax[0].imshow(or_pulse, clim=[0, 50], aspect='auto')
                ax[1].imshow(or_sin, clim=[0, 50], aspect='auto')
                
            fig.savefig(os.path.join(fig_save_path, probe+'_'+genotype+image_set+'_optotagged_response.png'))

### Image responsiveness over probes/genotypes/image sets
def calculate_sparseness(mean_response_vector):
    '''lifetime sparseness as used in marina's biorxiv paper (defined by Gallant)
    mean_response_vector (len n) should contain the trial mean of a cell's response 
    (however defined) over n conditions
    
    for population sparseness, mean_response_vector should be the mean stimulus 
    response of a population of n neurons
    '''
    
    sumsquared = float(np.sum(mean_response_vector)**2)
    sum_of_squares = float(np.sum(mean_response_vector**2))
    n = float(mean_response_vector.size)
    
    try:
        num = 1 - (1/n)*(sumsquared/sum_of_squares)
        denom = 1 - (1/n)
        
        ls = num/denom
    except:
        ls = np.nan
    
    return ls



def get_stimresponse_for_row(row, baseline_sub=True):
    
    crs = np.array(row['change_responses'])
    stim_response = np.mean(crs[:, 1030:1130], axis=1)
    if baseline_sub:
        stim_response = stim_response - np.mean(crs[:, 900:1000], axis=1)
    
    return stim_response

def get_stim_responsive_for_row(row):
    
    crs = np.array(row['change_responses'])
    stim_response = np.mean(crs[:, 1030:1130], axis=1)
    baseline_mean = np.mean(crs[:, 800:1000], axis=1)
    baseline_std = np.std(crs[:, 800:1000], axis=1)
    
    #return any([s>bm+5*bs for s,bm,bs in zip(stim_response, baseline_mean, baseline_std)])
    return np.array([s>bm+5*bs for s,bm,bs in zip(stim_response, baseline_mean, baseline_std)])
    

    
    
    #common_inds = common_ind_dict['_G'] if 'im012_r' in image_list else common_ind_dict['_H']
    #other_inds = other_ind_dict['_G'] if 'im012_r' in image_list else other_ind_dict['_H']
    
#    sp = []
#    for inds in [common_inds, other_inds]:
#        cr = np.mean(crs[inds], axis=0)
#        stim_response = np.mean(crs[inds, ])


fig, ax = plt.subplots()
for image_set in ['_G', '_H']:
    cr = np.mean(p_crmat_dict['B']['RS'][image_set][other_ind_dict[image_set]], axis=0)
    stim_response = np.mean(cr[:, 1050:1150], axis=1) - np.mean(cr[:, 900:1000], axis=1)
    h, b = np.histogram(stim_response, np.arange(50))
    h = h/np.sum(h)
    ax.step(b[:-1], h, where='post')
    #ax.hist(stim_response, bins=np.arange(-10, 50))
    print('mean', np.mean(stim_response))
    print(image_set, calculate_sparseness(stim_response)) 


combined_df['stim_response_baseline_sub'] = combined_df.apply(get_stimresponse_for_row, axis=1)
combined_df['stim_responsive'] = combined_df.apply(get_stim_responsive_for_row, axis=1)
combined_df['cell_class'] = ''

#ASSIGN CELL CLASSES TO CELLS
combined_df.loc[(combined_df['genotype']=='Vip-IRES-Cre;Ai32')&combined_df['optotagged'], 'cell_class'] = 'VIP'
combined_df.loc[(combined_df['genotype']=='Sst-IRES-Cre;Ai32')&combined_df['optotagged'], 'cell_class'] = 'SST'
combined_df.loc[(combined_df['duration']>=0.4)&(combined_df['optotagged']==False), 'cell_class'] = 'RS'
combined_df.loc[(combined_df['duration']<0.4)&(combined_df['optotagged']==False), 'cell_class'] = 'FS'

genotypes = ['Vip-IRES-Cre;Ai32', 'Sst-IRES-Cre;Ai32', 'RS', 'FS']
probe_grouped = combined_df.groupby('probe')
for probe, pgroup in probe_grouped:

    for genotype in genotypes:
        good_unit_filter = ((pgroup['snr']>1)&(pgroup['isi_viol']<1)&(pgroup['firing_rate']>0.1))
        gtoh_filter = (pgroup['mouseID']!='548722')
        if genotype in ['Vip-IRES-Cre;Ai32', 'Sst-IRES-Cre;Ai32']:
            df_filter = (
                        (pgroup['genotype']== genotype) &
                        (pgroup['optotagged']))
            
        elif genotype=='RS':
            df_filter = (
                         (pgroup['optotagged']==False) &
                         (pgroup['duration']>=0.4))
        elif genotype=='FS':
            df_filter = (
                         (pgroup['optotagged']==False) &
                         (pgroup['duration']<0.4))
            
        mouse_group = pgroup.loc[good_unit_filter&gtoh_filter&df_filter].groupby('mouseID')

        sparse = [[], []]
        frac_resp = [[],[]]
        for mid, mgroup in mouse_group:
            for imind, imageset in enumerate(['_G', '_H']):
                stim = mgroup.loc[mgroup['image_set'].str.contains(imageset), 'stim_response_baseline_sub']
                
                if len(stim)>0:
                    stim = np.array([np.mean(s[other_ind_dict[imageset]], axis=0) for s in stim])
                    s = calculate_sparseness(stim)
                else:
                    s = np.nan
                
                sparse[imind].append(s)
                
        fig, ax = plt.subplots()
        ax.plot(sparse[0], sparse[1], 'ko')
        ax.plot([0,1], [0,1], 'k--')
        ax.set_ylim([0,1])
        ax.set_xlim([0,1])
        ax.set_title(probe+' '+genotype)
        ax.set_aspect('equal')
        ax.text(0.01, 0.01, scipy.stats.wilcoxon(sparse[0], sparse[1])[1])
        #print(probe+' '+genotype, scipy.stats.wilcoxon(sparse[0], sparse[1]))

        
        g_frac_resp = pgroup.loc[good_unit_filter&gtoh_filter&
                                 df_filter&(pgroup['image_set'].str.contains('_G'))].groupby('mouseID').mean()['stim_responsive']
        h_frac_resp = pgroup.loc[good_unit_filter&gtoh_filter&
                                 df_filter&(pgroup['image_set'].str.contains('_H'))].groupby('mouseID').mean()['stim_responsive']
        
        if len(g_frac_resp)>0 and len(h_frac_resp)>0:
            frac_resp = pd.merge(g_frac_resp, h_frac_resp, on='mouseID')
            fig,ax = plt.subplots()
            ax.plot(frac_resp['stim_responsive_x'], frac_resp['stim_responsive_y'], 'ko')
            ax.set_ylim([0,1])
            ax.set_xlim([0,1])
            ax.plot([0,1], [0,1], 'k--')
            ax.set_title(probe+' '+genotype)
            ax.set_aspect('equal')
            
            
good_unit_filter = ((combined_df['snr']>1)&(combined_df['isi_viol']<1)&(combined_df['firing_rate']>0.1))
gtoh_filter = (combined_df['mouseID']!='548722')
new_g_order = np.array(['im036_r', 'im047_r', 'im012_r', 'im078_r', 'im044_r', 'im115_r', 'im083_r', 'im111_r', 'omitted'])
new_h_order = np.array(['im104_r', 'im114_r', 'im024_r', 'im034_r', 'im087_r', 'im005_r', 'im083_r', 'im111_r', 'omitted'])
new_g_inds = np.array([list(g_image_list).index(im) for im in new_g_order])
new_h_inds = np.array([list(h_image_list).index(im) for im in new_h_order])

for genotype in ('VIP', 'SST', 'RS', 'FS'):
    fig, ax = plt.subplots()
    fig.suptitle(genotype)
    for setind, (image_set, iminds)  in enumerate(zip(('_G', '_H'), (new_g_inds, new_h_inds))):
        
        fraction_responsive = combined_df.loc[good_unit_filter&gtoh_filter&
                                              (combined_df['cell_class']==genotype)&
                                              (combined_df['image_set'].str.contains(image_set))].get('stim_responsive')
        
#        crs = combined_df.loc[good_unit_filter&gtoh_filter&
#                                              (combined_df['cell_class']==genotype)&
#                                              (combined_df['image_set'].str.contains(image_set))].get('change_responses')
#        
        fraction_responsive = np.stack(fraction_responsive)[:, iminds]
        meanfr = np.mean(fraction_responsive, axis=0)
        private_order = np.append(np.argsort(meanfr[:-3])[::-1], [6,7,8])
        meanfr = meanfr[private_order]
        
        ax.plot(np.arange(6), meanfr[:-3], image_set_colors[setind]+'o', markerfacecolor='w', markersize=7)
        ax.plot(np.arange(6,8), meanfr[-3:-1], image_set_colors[setind]+'o', markersize=7)
        ax.plot(8, meanfr[-1], image_set_colors[setind]+'D', markersize=7)
        
        
        error = [scipy.stats.binom.interval(0.95, p=fr, n=fraction_responsive.shape[0]) for fr in meanfr]
        error = np.abs((np.array(error)/fraction_responsive.shape[0]) - meanfr[:, None])
        error[meanfr==0] = 0 #don't show error bar for no observations
        ax.errorbar(np.arange(9), meanfr, yerr=error.T, color=image_set_colors[setind], fmt='none')
        
        #ax.fill_between(np.arange(9), error[:, 0], error[:, 1], color=image_set_colors[setind], alpha=0.5)
    
    ax.set_xticks([2.5, 6.5, 8])
    ax.set_xticklabels(['private', 'common', 'omitted'])
    analysis.save_figure(fig, os.path.join(fig_save_path, 'image_responsiveness_newcolors\\allareas_' + genotype + '.pdf'))


probe_grouped = combined_df.groupby('probe')
for probe, pgroup in probe_grouped:
    good_unit_filter = ((pgroup['snr']>1)&(pgroup['isi_viol']<1)&(pgroup['firing_rate']>0.1))
    gtoh_filter = (pgroup['mouseID']!='548722')
    for genotype in ('VIP', 'SST', 'RS', 'FS'):
        
        fig, ax = plt.subplots()
        fig.suptitle(probe+' '+genotype)
        for setind, (image_set, iminds)  in enumerate(zip(('_G', '_H'), (new_g_inds, new_h_inds))):
            
            fraction_responsive = pgroup.loc[good_unit_filter&gtoh_filter&
                                                  (pgroup['cell_class']==genotype)&
                                                  (pgroup['image_set'].str.contains(image_set))].get('stim_responsive')
            
            if fraction_responsive.shape[0]>0:
                fraction_responsive = np.stack(fraction_responsive)[:, iminds]
                meanfr = np.mean(fraction_responsive, axis=0)
                private_order = np.append(np.argsort(meanfr[:-3])[::-1], [6,7,8])
                meanfr = meanfr[private_order]
                
                ax.plot(np.arange(6), meanfr[:-3], image_set_colors[setind]+'o', markerfacecolor='w', markersize=7)
                ax.plot(np.arange(6,8), meanfr[-3:-1], image_set_colors[setind]+'o', markersize=7)
                ax.plot(8, meanfr[-1], image_set_colors[setind]+'D', markersize=7)
                
                
                error = [scipy.stats.binom.interval(0.95, p=fr, n=fraction_responsive.shape[0]) for fr in meanfr]
                error = np.abs((np.array(error)/fraction_responsive.shape[0]) - meanfr[:, None])
                error[meanfr==0] = 0 #don't show error bar for no observations
                ax.errorbar(np.arange(9), meanfr, yerr=error.T, color=image_set_colors[setind], fmt='none')
                
            #ax.fill_between(np.arange(9), error[:, 0], error[:, 1], color=image_set_colors[setind], alpha=0.5)
        
        ax.set_xticks([2.5, 6.5, 8])
        ax.set_xticklabels(['private', 'common', 'omitted'])
        analysis.save_figure(fig, os.path.join(fig_save_path, 'image_responsiveness_newcolors\\'+probe+'_'+ genotype + '.pdf'))



def change_responses_over_time_group(group):
    sid = group.iloc[0]['sessionID']
    stim_table = combined_stim_df.loc[combined_stim_df['sessionID']==sid]
    change_times = stim_table.loc[(stim_table['change']==1)&(stim_table['active']), 'Start'].values
    change_time_quartiles =  np.digitize(change_times, [change_times.min() + t for t in np.array([0.25, 0.5, 0.75, 1])*(change_times.max() - change_times.min())])
    
    qcrs = group.apply(lambda row: change_responses_over_time_group_row(row, change_times, change_time_quartiles), axis=1)
    return qcrs


def change_responses_over_time_group_row(row, change_times, change_time_quartiles):
    qcrs = []
    for q in np.arange(4):
        qcts = change_times[change_time_quartiles==q]
        psth = analysis.makePSTH_numba(row['times'], qcts-1, 2)
        qcrs.append(psth[0])
    return np.array(qcrs)


def change_responses_over_time(row):
    sid = row['sessionID']
    stim_table = combined_stim_df.loc[combined_stim_df['sessionID']==sid]
    change_times = stim_table.loc[(stim_table['change']==1)&(stim_table['active']), 'Start'].values
    change_time_quartiles =  np.digitize(change_times, [change_times.min() + t for t in np.array([0.25, 0.5, 0.75, 1])*(change_times.max() - change_times.min())])
    
    qcrs = []
    for q in np.arange(4):
        qcts = change_times[change_time_quartiles==q]
        psth = analysis.makePSTH_numba(row['times'], qcts-1, 2)
        qcrs.append(psth[0])
    return np.array(qcrs)
        

good_unit_filter = ((combined_df['snr']>1)&(combined_df['isi_viol']<1)&(combined_df['firing_rate']>0.1)&(combined_df['presence_ratio']>0.98))
gtoh_filter = (combined_df['mouseID']!='548722')
for genotype in genotypes:
    
    for image_set in ('_G', '_H'):
        pgrouped = combined_df.loc[good_unit_filter&gtoh_filter&
                         (combined_df['cell_class']==genotype)&
                         (combined_df['image_set'].str.contains(image_set))].groupby('probe')
    
        for probe, pgroup in pgrouped:
        
            qcr = pgroup.groupby('sessionID').apply(change_responses_over_time_group)
            if qcr.shape[0]==1:
                qcr_stack = qcr.iloc[0]
            else:
                qcr_stack = np.stack(qcr)
            fig, ax = plt.subplots()
            fig.suptitle(probe+' '+genotype+image_set)
            ax.plot(np.mean(qcr_stack, axis=0).T)
            analysis.save_figure(fig, os.path.join(fig_save_path, 'cr_over_time\\' + probe+'_'+genotype+image_set+'_crbytimequartile.png'))
        
        
depths = combined_df.groupby('sessionID').position.transform(lambda x: np.median(np.stack(x)[:, 1]) - np.stack(x)[:, 1])
combined_df['depths'] = depths

### spike width hists
fig, axes = plt.subplots(5,1)
colors = ['k', 'k', 'r', 'g', 'purple']
for ind, (ax, color, genotype) in enumerate(zip(axes, colors, ['all', 'RS', 'FS', 'SST', 'VIP'])):
    bins = np.linspace(0, 1, 73)
    binwidth = bins[1]-bins[0]
    if genotype=='all':
        h,b = np.histogram(combined_df.loc[good_unit_filter&combined_df['genotype'].str.contains('Sst')].duration, bins=bins)
        alpha=1
    else:
        h,b = np.histogram(combined_df.loc[good_unit_filter&combined_df['genotype'].str.contains('Sst')&(combined_df['cell_class']==genotype)].duration, bins=bins)
        alpha=0.5
    ax.bar(b[:-1], h, width=binwidth, color=color, alpha=alpha)
    
    if ind==len(axes)-1:
        spinestohide=None
        xlabel = 'trough to peak (ms)'
        ylabel = 'unit count'
    else:
        spinestohide = ['top', 'bottom', 'right']
        ax.set_xticks([])
        xlabel=None
        ylabel=None
    formataxes(ax, xLabel=xlabel, yLabel=ylabel, spinesToHide=spinestohide)
    ax.text(0, ax.get_ylim()[1]*0.5, genotype, fontdict={'color':color, 'size':12})
analysis.save_figure(fig, os.path.join(fig_save_path, 'spikewidths_by_genotype_inSSTmice.png'))  


### plot example FS and RS waveforms
fig, ax = plt.subplots()
t = ctx_df.loc[ctx_df['duration']<0.3].iloc[2]['template']
minchan = np.unravel_index(np.argmin(t), t.shape)
ax.plot(t[:, minchan[1]], 'r', linewidth=2)
formataxes(ax, no_spines=True)

t = ctx_df.loc[ctx_df['duration']>0.45].iloc[3]['template']
minchan = np.unravel_index(np.argmin(t), t.shape)
ax.plot(t[:, minchan[1]], 'k', linewidth=2)
formataxes(ax, no_spines=True)
analysis.save_figure(fig, os.path.join(fig_save_path, 'example_FS_RS_waveforms.png'))


novelty = []
nerror = []
context = []
cerror = []
for probe in ['C']:
    for genotype in ['RS', 'FS', 'Vip-IRES-Cre;Ai32', 'Sst-IRES-Cre;Ai32']:
        
        h_cr = np.array(p_crmat_dict[probe][genotype]['_H']['cr'])
        g_cr = np.array(p_crmat_dict[probe][genotype]['_G']['cr'])
        
        stimmean = lambda x: np.mean(np.mean(x, axis=0)[:, 1050:1150], axis=1)
        stimsem =  lambda x: np.std(np.mean(np.mean(x, axis=0)[:, 1050:1150], axis=1))/x.shape[1]**0.5
        nov = np.mean(stim(h_cr[other_ind_dict['_H']]) - stim(h_cr[common_ind_dict['_H']])
        con = stim(h_cr[common_ind_dict['_H']]) - stim(g_cr[common_ind_dict['_G']])
        
        novelty.append(nov)
        context.append(con)
        nerror.append(stimsem())

fig, ax = plt.subplots(1,2)
ax[0].plot(novelty, 'ko', ms=8)
ax[1].plot(context, 'ko', ms=8)

[a.set_xticks(np.arange(4)) for a in ax]
[a.set_xticklabels(['RS', 'FS', 'VIP', 'SST']) for a in ax]
ax[0].set_title('Novelty')
ax[0].axhline(0, color='k', linestyle='dotted')
ax[1].set_title('Context')
ax[0].set_ylabel('Population Firing Rate Modulation')
[formataxes(a) for a in ax]


fig, ax = plt.subplots()
ax.plot(novelty, 'ko', ms=8)
ax.plot(context, 'ko', ms=8, markerfacecolor='w')

ax.set_xticks(np.arange(4))
ax.set_xticklabels(['RS', 'FS', 'VIP', 'SST'])
ax.axhline(0, color='k', linestyle='dotted')
ax.set_title('Novelty and Context modulation in V1')
ax.set_ylabel('Population Firing Rate Modulation')
formataxes(ax)
ax.legend(['Novelty', 'Context'])




# probe spike rates across all experiments
good_unit_filter = ((combined_df['snr']>1)&(combined_df['isi_viol']<1)&(combined_df['firing_rate']>0.1)&(combined_df['presence_ratio']>0.98))
gtoh_filter = (combined_df['mouseID']!='548722')
flatten = lambda l: [item[0] for sublist in l for item in sublist]
bins = np.arange(9000)
session_grouped = combined_df.groupby('sessionID')
ash = {'RS':[], 'FS':[], 'VIP':[], 'SST':[]}
for cellclass in ash:
            
    grouptimes = flatten(combined_df[good_unit_filter&gtoh_filter&(combined_df['cell_class']==cellclass)]['times'])
    
    h,b = np.histogram(grouptimes, bins=bins)
    ash[cellclass] = h
        
for g in ash:
    fig, ax = plt.subplots()
    ax.plot(bins[:-1], ash[g], 'k')
    ax.axvline(3600)
    ax.axvline(5130, color='g')
    ax.set_title(g)



bins = np.arange(9000)
VIP_ash = {'RS':[], 'FS':[], 'VIP':[], 'SST':[]}
for cellclass in ash:
            
    grouptimes = flatten(combined_df[good_unit_filter&gtoh_filter&
                                     (combined_df['cell_class']==cellclass)&
                                     (combined_df['genotype'].str.contains('Sst'))&
                                     (combined_df['image_set'].str.contains('_G'))]['times'])
    
    h,b = np.histogram(grouptimes, bins=bins)
    VIP_ash[cellclass] = h
        
for g in ash:
    fig, ax = plt.subplots()
    ax.plot(bins[:-1], VIP_ash[g], 'k')
    ax.axvline(3600)
    ax.axvline(5130, color='g')
    ax.set_title(g)



early_satiated = ['554013', '536211', '556016', '550324', '540536']
late_satiated=['556014', '548721', '544835', '544836', '546507', '546508']

bins = np.arange(9000)
early_ash = {'RS':[], 'FS':[], 'VIP':[], 'SST':[]}
late_ash = {'RS':[], 'FS':[], 'VIP':[], 'SST':[]}
for cellclass in ash:
    for sate_group, sate_dict in zip([early_satiated, late_satiated], [early_ash, late_ash]):        
        grouptimes = flatten(combined_df[good_unit_filter&gtoh_filter&
                                         (combined_df['cell_class']==cellclass)&
                                         (combined_df['image_set'].str.contains('_G'))&
                                         (combined_df['mouseID'].isin(sate_group))]['times'])
        
        h,b = np.histogram(grouptimes, bins=bins)
        sate_dict[cellclass] = h

colors = ['k', 'r']    
for g in early_ash:
    fig, ax = plt.subplots()
    ax.set_title(g)

    for color, sate_dict in zip(colors, [early_ash, late_ash]):
        ax.plot(bins[:-1], np.convolve(sate_dict[g]/sate_dict[g].mean(), np.ones(10), 'same'), color)
    
    ax.axvline(3600)
    ax.axvline(5130, color='g')

g_early_runners = ['532246', '540536', '545996', '556014']

bins = np.arange(9000)
early_run_ash = {'RS':[], 'FS':[], 'VIP':[], 'SST':[]}
for cellclass in ash:
           
    grouptimes = flatten(combined_df[good_unit_filter&gtoh_filter&
                                     (combined_df['cell_class']==cellclass)&
                                     (combined_df['image_set'].str.contains('_G'))&
                                     (combined_df['mouseID'].isin(g_early_runners))]['times'])
    
    h,b = np.histogram(grouptimes, bins=bins)
    early_run_ash[cellclass] = h

colors = ['k', 'r']    
for g in early_run_ash:
    fig, ax = plt.subplots()
    ax.set_title(g)

    ax.plot(bins[:-1], np.convolve(early_run_ash[g]/early_run_ash[g].mean(), np.ones(10), 'same'), color)
    
    ax.axvline(3600)
    ax.axvline(5130, color='g')

