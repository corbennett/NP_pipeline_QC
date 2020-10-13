# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:18:35 2020

@author: svc_ccg
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import os, json, shutil
import analysis
from numba import njit
import visual_behavior
from get_sessions import glob_file
import ecephys
from probeSync_qc import get_sync_line_data
import probeSync_qc as probeSync
import cv2
import pandas as pd
import plotly
import plotly.tools as tls

probe_color_dict = {'A': 'orange',
                        'B': 'r',
                        'C': 'k',
                        'D': 'g',
                        'E': 'b',
                        'F': 'm'}

def find_spikes_per_trial(spikes, trial_starts, trial_ends):
    tsinds = np.searchsorted(spikes, trial_starts)
    teinds = np.searchsorted(spikes, trial_ends)
    
    return teinds - tsinds

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

def makePSTH(spikes,startTimes,windowDur,binSize=0.01, avg=True):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros((len(startTimes),bins.size-1))    
    for i,start in enumerate(startTimes):
        counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,bins)[0]
    if avg:
        counts = counts.mean(axis=0)
    counts /= binSize
    return counts


def plot_rf(mapping_pkl_data, spikes, first_frame_offset, frameAppearTimes, resp_latency=0.025, plot=True, returnMat=False):

    
    rfFlashStimDict = mapping_pkl_data
    rfStimParams = rfFlashStimDict['stimuli'][0] 
    rf_pre_blank_frames = int(rfFlashStimDict['pre_blank_sec']*rfFlashStimDict['fps'])
    first_rf_frame = first_frame_offset + rf_pre_blank_frames
    rf_frameTimes = frameAppearTimes[first_rf_frame:]
    rf_trial_start_times = rf_frameTimes[np.array([f[0] for f in np.array(rfStimParams['sweep_frames'])]).astype(np.int)]

    #extract trial stim info (xpos, ypos, ori)
    sweep_table = np.array(rfStimParams['sweep_table'])   #table with rfstim parameters, indexed by sweep order to give stim for each trial
    sweep_order = np.array(rfStimParams['sweep_order'])   #index of stimuli for sweep_table for each trial
    
    trial_xpos = np.array([pos[0] for pos in sweep_table[sweep_order, 0]])
    trial_ypos = np.array([pos[1] for pos in sweep_table[sweep_order, 0]])
    trial_ori = sweep_table[sweep_order, 3]
    
    xpos = np.unique(trial_xpos)
    ypos = np.unique(trial_ypos)
    ori = np.unique(trial_ori)
    
    respInds = tuple([(np.where(ypos==y)[0][0], np.where(xpos==x)[0][0], np.where(ori==o)[0][0]) for (y,x,o) in zip(trial_ypos, trial_xpos, trial_ori)])
    trial_spikes = find_spikes_per_trial(spikes, rf_trial_start_times+resp_latency, rf_trial_start_times+resp_latency+0.2)
    respMat = np.zeros([ypos.size, xpos.size, ori.size])
    for (respInd, tspikes) in zip(respInds, trial_spikes):
        respMat[respInd] += tspikes
    
#    bestOri = np.unravel_index(np.argmax(respMat), respMat.shape)[-1]

    return respMat


def plot_psth_change_flashes(change_times, spikes, preTime = 0.05, postTime = 0.55, sdfSigma=0.005):
    
    sdf, t = makePSTH_numba(spikes,change_times-preTime,preTime+postTime, convolution_kernel=sdfSigma*2)
    
    return sdf, t


def lickTriggeredLFP(lick_times, lfp, lfp_time, agarChRange=None, num_licks=20, windowBefore=0.5, windowAfter=0.5, min_inter_lick_time = 0.5, behavior_duration=3600): 
    
    first_lick_times = lick_times[lick_times>lfp_time[0]+windowBefore]
    first_lick_times = first_lick_times[:np.min([len(first_lick_times), num_licks])]
    
    probeSampleRate = 1./np.median(np.diff(lfp_time))
    samplesBefore = int(round(windowBefore * probeSampleRate))
    samplesAfter = int(round(windowAfter * probeSampleRate))
    
    last_lick_ind = np.where(lfp_time<=first_lick_times[-1])[0][-1]
    lfp = lfp[:last_lick_ind+samplesAfter+1]
    lfp = lfp - np.mean(lfp, axis=0)[None, :]    

    if agarChRange is not None:
        agar = np.median(lfp[:,agarChRange[0]:agarChRange[1]],axis=1)
        lfp = lfp-agar[:,None]
    
    lickTriggeredAv = np.full([first_lick_times.size, samplesBefore+samplesAfter, lfp.shape[1]], np.nan)
    lick_inds = np.searchsorted(lfp_time, first_lick_times)
    for il, li in enumerate(lick_inds):
       lickTriggeredAv[il, :, :] = lfp[li-samplesBefore:li+samplesAfter] 


    m = np.nanmean(lickTriggeredAv, axis=0)*0.195  #convert to uV
    mtime = np.linspace(-windowBefore, windowAfter, m.size)  
    return m, mtime, first_lick_times


def plot_lick_triggered_LFP(lfp_dict, lick_times, FIG_SAVE_DIR, prefix='', 
                            agarChRange=None, num_licks=20, windowBefore=0.5, 
                            windowAfter=1.5, min_inter_lick_time = 0.5, behavior_duration=3600):
    
    for p in lfp_dict:

        plfp = lfp_dict[p]['lfp']
        lta, ltime, first_lick_times = analysis.lickTriggeredLFP(lick_times, plfp, lfp_dict[p]['time'], 
                                               agarChRange=[325, 350], num_licks=20, windowBefore=windowBefore,
                                               windowAfter=windowAfter, min_inter_lick_time=0.5)
        
        fig, axes = plt.subplots(2,1)
        fig.suptitle(p + ' Lick-triggered LFP, ' + str(len(first_lick_times)) + ' rewarded lick bouts')
        axes[0].imshow(lta.T, aspect='auto')
        axes[1].plot(np.mean(lta, axis=1), 'k')
        for a in axes:
            a.set_xticks(np.arange(0, windowBefore+windowAfter, windowBefore)*2500)
            a.set_xticklabels(np.round(np.arange(-windowBefore, windowAfter, windowBefore), decimals=2))
            
        axes[1].set_xlim(axes[0].get_xlim())
        axes[0].tick_params(bottom=False, labelbottom=False)
        axes[1].set_xlabel('Time from lick bout (s)')
        axes[0].set_ylabel('channel')
        axes[1].set_ylabel('Mean across channels')
        
        save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix+'Probe' + p + ' lick-triggered LFP'))


def get_first_lick_times(lick_times, min_inter_lick_time=0.5, rewarded=True):
    
    first_lick_times = lick_times[np.insert(np.diff(lick_times)>=min_inter_lick_time, 0, True)]
    
    return first_lick_times


def get_rewarded_lick_times(lickTimes, frameTimes, trials, min_inter_lick_time=0.5):
    
    trial_start_frames = np.array(trials['startframe'])
    trial_end_frames = np.array(trials['endframe'])
    trial_start_times = frameTimes[trial_start_frames]
    trial_end_times = frameTimes[trial_end_frames]
    
    first_lick_times = lickTimes[np.insert(np.diff(lickTimes)>=min_inter_lick_time, 0, True)]
    first_lick_trials = get_trial_by_time(first_lick_times, trial_start_times, trial_end_times)
    
    hit = np.array(trials['response_type']=='HIT')
    
    hit_lick_times = first_lick_times[np.where(hit[first_lick_trials])[0]]

    return hit_lick_times


def get_trial_by_time(times, trial_start_times, trial_end_times):
    trials = []
    for time in times:
        if trial_start_times[0]<=time<trial_end_times[-1]:
            trial = np.where((trial_start_times<=time)&(trial_end_times>time))[0][0]
        else:
            trial = -1
        trials.append(trial)
    
    return np.array(trials)


def vectorize_edgetimes(on_times, off_times, sampleperiod = 0.001):
    
    on_times_samp = np.round(on_times/sampleperiod, 0).astype(int)
    off_times_samp = np.round(off_times/sampleperiod, 0).astype(int)
    
    last_time = np.max([on_times_samp.max(), off_times_samp.max()])
    vector = np.zeros(int(last_time))
    times = np.arange(len(vector))*sampleperiod
    
    
    if off_times_samp[0] < on_times_samp[0]:
        on_times_samp = np.insert(on_times_samp, 0, 0)
    
    if on_times[-1]>off_times[-1]:
        off_times_samp = np.append(off_times_samp, int(last_time))
    
    on_intervals = [slice(on, off) for on, off in zip(on_times_samp, off_times_samp)]
    #off_intervals = [slice(off, on) for on, off in zip(on_times[1:], off_times)]
    
    for interval in on_intervals:
        vector[interval] = 1
            
    return vector, times

def plot_vsync_and_diode(syncDataset, FIG_SAVE_DIR, prefix=''):
    
    monitor_lag = get_monitor_lag(syncDataset)
    dioder, diodef = probeSync.get_diode_times(syncDataset)
    vf = probeSync.get_vsyncs(syncDataset)
    
    start_session_diode_vector, dtimes = vectorize_edgetimes(dioder[:10], diodef[:10])
    
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches([15, 6])
    fig.suptitle('vsync/diode alignment')
    ax[0].plot(dtimes, start_session_diode_vector, 'k')
    ax[0].plot(vf[:500], 0.5*np.ones(500), 'r|', ms=20)
    ax[0].plot(vf[60], 0.5, 'g|', ms=30)
    ax[0].set_xlim([vf[0]-1, vf[60]+0.5])
    ax[0].plot([vf[60], vf[60]+monitor_lag], [0.75,0.75], 'b-')
    ax[0].set_xlabel('Experiment time (s)')
    
    ax[1].plot(dtimes, start_session_diode_vector, 'k')
    ax[1].plot(vf[50:70], 0.5*np.ones(20), 'r|', ms=20)
    ax[1].plot(vf[60], 0.5, 'g|', ms=30)
    ax[1].set_xlim([vf[50], vf[70]])
    ax[1].plot([vf[60], vf[60]+monitor_lag], [0.75,0.75], 'b-')
    ax[1].set_xlabel('Experiment time (s)')
    
    ax[1].legend(['diode', 'vf', 'frame 60', 'lag'], markerscale=0.5)

    save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix+'vsync_with_diode.png'))


def get_monitor_lag(syncDataset):

    dioder, diodef = probeSync.get_diode_times(syncDataset)
    vf = probeSync.get_vsyncs(syncDataset)
    
    lag = np.min([np.min(np.abs(d-vf[60])) for d in [diodef, dioder]])
    
    return lag


def plot_frame_intervals(vsyncs, behavior_frame_count, mapping_frame_count, 
                         behavior_start_frame, mapping_start_frame, 
                         replay_start_frame, save_dir, prefix=''):
    
    fig, ax = plt.subplots()
    fig.suptitle('stim frame intervals')
    ax.plot(np.diff(vsyncs))
    ax.set_ylim([0, 0.2])
    vline_locs = [behavior_start_frame, mapping_start_frame, 
                  replay_start_frame, replay_start_frame+behavior_frame_count]
    for v in vline_locs:
        ax.axvline(v, color='k', linestyle='--')

    
    ax.set_xlabel('frames')
    ax.set_ylabel('interval, s (capped at 0.2)')
    
    ax.text(behavior_start_frame + behavior_frame_count/2, 0.15, 
            'behavior', horizontalalignment='center')
    ax.text(mapping_start_frame+mapping_frame_count/2, 0.15, 
            'rf', horizontalalignment='center')
    ax.text(replay_start_frame+behavior_frame_count/2, 0.15, 'replay', horizontalalignment='center')
    
    save_figure(fig, os.path.join(save_dir, prefix+'stim_frame_intervals.png'))


def plot_vsync_interval_histogram(vf, FIG_SAVE_DIR, prefix=''):
    
    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle('Vsync interval histogram')
    
    bins = np.arange(12, 40, 0.1)
    ax.hist(np.diff(vf)*1000, bins=bins)
    v = ax.axvline(16.667, color='k', linestyle='--')
    ax.set_ylabel('number of frame intervals')
    ax.set_xlabel('frame interval (ms)')
    ax.legend([v], ['expected interval'])
    
    save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix+'vsync_interval_histogram.png'))


def vsync_report(syncDataset, total_pkl_frames, FIG_SAVE_DIR, prefix=''):
    
    vf = probeSync.get_vsyncs(syncDataset)
    
    report = {}
    intervals = np.diff(vf)
    report['sync_vsync_frame_count'] = len(vf)
    report['pkl frame count'] = total_pkl_frames
    report['sync_pkl_framecount_match'] = 'TRUE' if len(vf)==total_pkl_frames else 'FALSE'
    report['mean interval'] = intervals.mean()
    report['median interval'] = np.median(intervals)
    report['std of interval'] = intervals[intervals<1].std()
    
    report['num dropped frames'] = int(np.sum(intervals>0.025))-2
    report['num intervals 0.1 <= x < 1'] = int(np.sum((intervals<1)&(intervals>=0.1)))
    report['num intervals >= 1 (expected = 2)'] = int(np.sum(intervals>=1))
    report['monitor lag'] = get_monitor_lag(syncDataset)
    
    save_json(report, os.path.join(FIG_SAVE_DIR, prefix+'vsync_report.json'))
    
    
def plot_population_change_response(probe_dict, behavior_start_frame, replay_start_frame, 
                                    trials, FRAME_APPEAR_TIMES, FIG_SAVE_DIR, ctx_units_percentile=66, prefix=''):
    
    
    change_frames = np.array(trials['change_frame'].dropna()).astype(int)+1
    active_change_times = FRAME_APPEAR_TIMES[change_frames+behavior_start_frame]
    passive_change_times = FRAME_APPEAR_TIMES[change_frames+replay_start_frame]
    
    lfig, lax = plt.subplots()
    preTime = 0.05
    postTime = 0.55
    for p in probe_dict:
        
        u_df = probe_dict[p]
        good_units = u_df[(u_df['quality']=='good')&(u_df['snr']>1)]
    #    max_chan = good_units['peak_channel'].max()
    #    # take spikes from the top n channels as proxy for cortex
    #    spikes = good_units.loc[good_units['peak_channel']>max_chan-num_channels_to_take_from_top]['times']
        ctx_bottom_chan = np.percentile(good_units['peak_channel'], ctx_units_percentile)
        spikes = good_units.loc[good_units['peak_channel']>ctx_bottom_chan]['times']
        sdfs = [[],[]]
        for s in spikes:
            s = s.flatten()
            if s.size>3600:
                for icts, cts in enumerate([active_change_times, passive_change_times]): 
                    sdf,t = analysis.plot_psth_change_flashes(cts, s, preTime=preTime, postTime=postTime)
                    sdfs[icts].append(sdf)
     
        # plot population change response
        fig, ax = plt.subplots()
        title = p + ' population change response'
        fig.suptitle(title)
        ax.plot(t, np.mean(sdfs[0], axis=0), 'k')
        ax.plot(t, np.mean(sdfs[1], axis=0), 'g')
        ax.legend(['active', 'passive'])
        ax.axvline(preTime, c='k')
        ax.axvline(preTime+0.25, c='k')
        ax.set_xticks(np.arange(0, preTime+postTime, 0.05))
        ax.set_xticklabels(np.round(np.arange(-preTime, postTime, 0.05), decimals=2))
        ax.set_xlabel('Time from change (s)')
        ax.set_ylabel('Mean population response')
        save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix + title + '.png'))
        #fig.savefig(os.path.join(FIG_SAVE_DIR, title + '.png'))
        
        mean_active = np.mean(sdfs[0], axis=0)
        mean_active_baseline = mean_active[:int(preTime*1000)].mean()
        baseline_subtracted = mean_active - mean_active_baseline
        lax.plot(t, baseline_subtracted/baseline_subtracted.max(), c=probe_color_dict[p])
        
    lax.legend(probe_dict.keys())
    lax.set_xlim([preTime, preTime+0.1])
    lax.set_xticks(np.arange(preTime, preTime+0.1, 0.02))
    lax.set_xticklabels(np.arange(0, 0.1, 0.02))
    lax.set_xlabel('Time from change (s)')
    lax.set_ylabel('Normalized response')
    save_figure(lfig, os.path.join(FIG_SAVE_DIR, prefix+'pop_change_response_latency_comparison.png'))
    #lfig.savefig(os.path.join(FIG_SAVE_DIR, 'pop_change_response_latency_comparison.png'))
    
    
    
def plot_running_wheel(behavior_data, mapping_data, replay_data, FIG_SAVE_DIR, prefix=''):   
    
    ### Plot Running Wheel Data ###    
    rfig, rax = plt.subplots(2,1)
    rfig.set_size_inches(12, 4)
    rfig.suptitle('Running')
    time_offset = 0
    colors = ['k', 'g', 'r']
    for ri, rpkl in enumerate([behavior_data, mapping_data, replay_data]):
        key = 'behavior' if 'behavior' in rpkl['items'] else 'foraging'
        intervals = rpkl['items']['behavior']['intervalsms'] if 'intervalsms' not in rpkl else rpkl['intervalsms']
        time = np.insert(np.cumsum(intervals), 0, 0)/1000.
        
        dx,vsig,vin = [rpkl['items'][key]['encoders'][0][rkey] for rkey in ('dx','vsig','vin')]
        
#        cum_dist = np.cumsum(dx)
#        cum_dist = cum_dist[:len(time)]
        
        run_speed = visual_behavior.analyze.compute_running_speed(dx[:len(time)],time,vsig[:len(time)],vin[:len(time)])
        cum_dist = np.cumsum(run_speed)*0.01667*0.01 # dist = speed*time then convert to meters
        rax[0].plot(time+time_offset, run_speed, colors[ri])
        rax[1].plot(time+time_offset, cum_dist, colors[ri])
        time_offset = time_offset + time[-1]
        
    #rax[0].set_xlabel('Time (s)')
    rax[0].set_ylabel('Run Speed (cm/s)')
    rax[0].legend(['behavior', 'rf map', 'passive'])
    
    rax[1].set_ylabel('Cumulative Distance (m)')
    rax[0].set_xlabel('Time (s)')
    
    save_figure(rfig, os.path.join(FIG_SAVE_DIR, prefix+'run_speed.png'))
    #rfig.savefig(os.path.join(FIG_SAVE_DIR, 'run_speed.png'))


def plot_unit_quality_hist(metrics_dict, FIG_SAVE_DIR, prefix=''):
    
    fig, ax = plt.subplots()
    legend_artist = []
    legend_label = []
    for ip, probe in enumerate(metrics_dict):
        p = metrics_dict[probe]
        
        labels = np.sort(np.unique(p['quality']))
        count = []
        colors = ['g', '0.4']
        bottom = 0
        
        for il, l in enumerate(labels):
            num_units = np.sum(p['quality']==l)
            count.append(num_units)
            
            if l not in legend_label:
                b = ax.bar(ip, num_units, bottom=bottom, color=colors[il])
                legend_artist.append(b)
                legend_label.append(l)
            else:
                ax.bar(ip, num_units, bottom=bottom, color=colors[il])
            bottom = np.cumsum(count)
            
        
    ax.set_xticks(np.arange(len(metrics_dict)))
    ax.set_xticklabels([p for p in metrics_dict])

    ax.legend(legend_artist, legend_label)
    ax.set_xlabel('Probe')
    ax.set_ylabel('Unit count')
    
    save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix+'unit_quality_hist.png'))
    #fig.savefig(os.path.join(FIG_SAVE_DIR, 'unit_quality_hist.png'))


def plot_unit_distribution_along_probe(metrics_dict, info_dict, FIG_SAVE_DIR, prefix=''):
    
    for ip, probe in enumerate(metrics_dict):
        p = metrics_dict[probe]
        good_units = p[p['quality']=='good']
        noise_units = p[p['quality']=='noise']
        
        fig, axes = plt.subplots(ncols=1, nrows=2, constrained_layout=True, sharex=True,
                             gridspec_kw={'width_ratios':[1], 'height_ratios':[1,6]})
        fig.suptitle('Probe {} unit distribution'.format(probe))
        
        bins = np.arange(0, 384, 20)
        goodhist = axes[1].hist(good_units['peak_channel'], bins=bins, color='g')
        noisehist = axes[1].hist(noise_units['peak_channel'], bins=bins, color='k', alpha=0.8)
        
        mask = np.array(info_dict[probe]['mask']).astype(int)
        axes[0].plot(np.arange(384), mask, 'k')
        axes[0].legend(['mask'])
        axes[0].axis('off')
        
        axes[1].set_xlabel('peak channel')
        axes[1].set_ylabel('unit count')
        
        axes[1].legend([goodhist[2][0], noisehist[2][0]], ['good', 'noise'])
        
        save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix+'Probe_{}_unit_distribution.png'.format(probe)))
        #fig.savefig(os.path.join(FIG_SAVE_DIR, 'Probe_{}_unit_distribution.png'.format(probe)))


def probe_yield_report(metrics_dict, info_dict, FIG_SAVE_DIR, prefix=''):
    
    report = {p:{} for p in metrics_dict}
    for probe in metrics_dict:
        p = metrics_dict[probe]
        report[probe]['num_good_units'] = int(np.sum(p['quality']=='good'))
        report[probe]['num_noise_units'] = int(np.sum(p['quality']=='noise'))
        
        info = info_dict[probe]
        report[probe]['surface_channel'] = info['surface_channel']
        report[probe]['air_channel'] = info['air_channel']
        report[probe]['num_masked_channels'] = int(np.sum(info['mask']))
  
    save_json(report, os.path.join(FIG_SAVE_DIR, prefix+'probe_yield_report.json'))
    

def plot_all_spike_hist(probe_dict, FIG_SAVE_DIR, prefix=''):
    
    flatten = lambda l: [item[0] for sublist in l for item in sublist]

    for p in probe_dict:
        u_df = probe_dict[p]
        good_units = u_df[(u_df['quality']=='good')&(u_df['snr']>1)]
        
        spikes = flatten(good_units['times'].to_list())
        binwidth = 1
        bins = np.arange(0, np.max(spikes), binwidth)
        hist, bin_e = np.histogram(spikes, bins)
        
        fig, ax = plt.subplots()
        fig.suptitle('spike histogram (good units), Probe ' + p)
        ax.plot(bin_e[1:-1], hist[1:])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Spike Count per ' + str(binwidth) + ' second bin')
        save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix+'Probe' + p + ' spike histogram'))
    
#    for ip, probe in enumerate(probe_dirs):
#        p_name = probe.split('_')[-2][-1]
#        base = os.path.join(os.path.join(probe, 'continuous'), 'Neuropix-PXI-100.0')
#        times_file = glob_file(base, 'spike_times.npy')
#        if times_file is not None:
#            times = np.load(times_file)
#            times = times/30000.
#            
#            fig, ax = plt.subplots()
#            bins = np.arange(0, times.max(), 1)
#            hist, b = np.histogram(times, bins=bins)
#            ax.plot(b[:-1], hist, 'k')
#            
#            fig.suptitle('Probe {} all spike time histogram'.format(p_name))
#            save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix+'Probe_{}_all_spike_time_hist.png'.format(p_name)))


def plot_barcode_interval_hist(probe_dirs, syncDataset, FIG_SAVE_DIR, prefix=''):
    
    for ip, probe in enumerate(probe_dirs):
        p_name = probe.split('_')[-2][-1]        
        base = os.path.join(probe, r'events\\Neuropix-PXI-100.0\\TTL_1')
        
        channel_states_file = glob_file(base, 'channel_states.npy')
        event_times_file = glob_file(base, 'event_timestamps.npy')
        
        if channel_states_file and event_times_file:
            
            #get barcode intervals from probe events file                        
            channel_states = np.load(channel_states_file)
            event_times = np.load(event_times_file)
            
            beRising = event_times[channel_states>0]/30000.
            beFalling = event_times[channel_states<0]/30000.
            be_t, be = ecephys.extract_barcodes_from_times(beRising, beFalling)
            
            barcode_intervals = np.diff(be_t)
            
            #get intervals from sync file for comparison
            bRising, bFalling = get_sync_line_data(syncDataset, channel=0)
            bs_t, bs = ecephys.extract_barcodes_from_times(bRising, bFalling)
            
            sync_barcode_intervals = np.diff(bs_t)
            
            fig, ax = plt.subplots(1,2)
            fig.set_size_inches([8, 4])
            bins = np.arange(np.min([barcode_intervals.min(), sync_barcode_intervals.min()])-1, 
                             np.max([barcode_intervals.max(), sync_barcode_intervals.max()])+1)

            ax[0].hist(barcode_intervals, bins)
            ax[1].hist(sync_barcode_intervals, bins)
            ax[0].axhline(len(be_t)-1)
            ax[1].axhline(len(bs_t)-1)
            
            ax[0].set_ylim([0.1, len(be_t)])
            ax[1].set_ylim([0.1, len(bs_t)])
            
            [a.set_yscale('log') for a in ax]
            
            
            ax[0].set_title('Probe {} ephys barcode intervals'.format(p_name))
            ax[1].set_title('Probe {} sync barcode intervals'.format(p_name))
            
            save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix+'Probe_{}_barcode_interval_hist.png'.format(p_name)))

def plot_barcode_intervals(probe_dirs, syncDataset, FIG_SAVE_DIR, prefix=''):
    
    fig, ax = plt.subplots()
    fig.suptitle('Sync Barcode Intervals')
    bs_t, bs = probeSync.get_sync_barcodes(syncDataset)
    ax.plot(np.diff(bs_t), 'k')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    pfig, pax = plt.subplots(1,2)
    pfig.set_size_inches([8,4])
    pfig.suptitle('Probe Barcode Intervals')
    for ip, probe in enumerate(probe_dirs):
        
        p_name = probe.split('_')[-2][-1]     
        be_t, be = probeSync.get_ephys_barcodes(probe)
        shift, p_sampleRate, m_endpoints = ecephys.get_probe_time_offset(bs_t, bs, be_t, be, 0, 30000)
        
        pax[0].plot(np.diff(be_t), probe_color_dict[p_name])
        pax[0].set_title('uncorrected')
        pax[1].plot(np.diff(be_t)*(30000./p_sampleRate), probe_color_dict[p_name])
        pax[1].set_title('corrected')
    
    pax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    pax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    pax[0].legend([probe.split('_')[-2][-1] for probe in probe_dirs])
    
    save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix+'Sync_barcode_intervals.png'))
    save_figure(pfig, os.path.join(FIG_SAVE_DIR, prefix+'Probe_barcode_intervals.png'))
    

def plot_barcode_matches(probe_dirs, syncDataset, FIG_SAVE_DIR, prefix=''):
    
    bs_t, bs = probeSync.get_sync_barcodes(syncDataset)
    
    fig, ax = plt.subplots()
    yticks=[]
    for ip, probe in enumerate(probe_dirs):
        
        p_name = probe.split('_')[-2][-1]     
        yticks.append(p_name)
        
        be_t, be = probeSync.get_ephys_barcodes(probe)
        match = np.zeros(len(bs))
        for ib, b in enumerate(bs):
            if b in be:
                match[ib]=1
        
        matches = np.where(match==1)[0]
        non_matches = np.where(match==0)[0]
        ax.plot(matches, np.ones(len(matches))*ip, color = 'g', marker='|', markersize=12, markeredgewidth=3, linewidth=0)
        ax.plot(non_matches, np.ones(len(non_matches))*ip, color = 'r', marker='|', markersize=12, markeredgewidth=5,linewidth=0)
    
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(yticks)
    ax.set_xlabel('Barcode Number')
    ax.set_ylabel('Probe')
    
    green_patch = mpatches.Patch(color='green', label='match')
    red_patch = mpatches.Patch(color='red', label='no match')
    ax.legend(handles=[green_patch, red_patch])
    save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix+'barcode_matching.png'))
    
    
def probe_sync_report(probe_dirs, syncDataset, FIG_SAVE_DIR, prefix=''):
 
    bs_t, bs = probeSync.get_sync_barcodes(syncDataset)
 
    alignment_dict = {}    
    for ip, probe in enumerate(probe_dirs):
        
        p_name = probe.split('_')[-2][-1]     
        alignment_dict[p_name] = {}
        
        be_t, be = probeSync.get_ephys_barcodes(probe)
        shift, p_sampleRate, m_endpoints = ecephys.get_probe_time_offset(bs_t, bs, be_t, be, 0, 30000)
        
        alignment_dict[p_name]['shift'] = np.float(shift)
        alignment_dict[p_name]['sample_rate'] = np.float(p_sampleRate)

    save_file = os.path.join(FIG_SAVE_DIR, prefix+'probe_sync_registration.json')
    save_json(alignment_dict, save_file)



def lost_camera_frame_report(paths, FIG_SAVE_DIR, prefix='', cam_report_keys=None):
    
    if cam_report_keys is None:
        cam_report_keys = [('RawBehaviorTrackingVideoMetadata', 'Behavior'),
                           ('RawEyeTrackingVideoMetadata', 'Eye'),
                           ('RawFaceTrackingVideoMetadata', 'Face')]
    
    report = {}
    for cam, name in cam_report_keys:
        if cam in paths:
            cam_meta = read_json(paths[cam])
            cam_meta = cam_meta['RecordingReport']
            report[name] = {}
            
            lost = cam_meta['FramesLostCount']
            recorded = cam_meta['FramesRecorded']
            
            report[name]['lost frame count'] = lost
            report[name]['recorded frame count'] = recorded
            report[name]['percent lost'] = 100*lost/(lost+recorded)

    save_file = os.path.join(FIG_SAVE_DIR, prefix+'cam_frame_report.json')
    save_json(report, save_file)
    

def camera_frame_grabs(paths, syncDataset, FIG_SAVE_DIR, epoch_start_times, epoch_end_times, epoch_frame_nums = [4,2,4], prefix='', cam_video_keys=None):
    
    if cam_video_keys is None:
        cam_video_keys = [('RawBehaviorTrackingVideo', 'Behavior', 'beh_frame_received'),
                           ('RawEyeTrackingVideo', 'Eye', 'eye_frame_received'),
                           ('RawFaceTrackingVideo', 'Face', 'face_frame_received')]
    
    videos_present = [c for c in cam_video_keys if c[0] in paths]
    num_videos = len(videos_present)
    
    #get frames spanning the 3 script epochs
    frames_to_grab = get_frames_from_epochs(videos_present, syncDataset, 
                                            epoch_start_times, epoch_end_times, epoch_frame_nums)
    
    cumulative_epoch_frames = np.cumsum(epoch_frame_nums)
    epoch_colors = ['k', 'g', 'r']
    get_epoch = lambda x: np.where(cumulative_epoch_frames>x)[-1][0]
    
    fig = plt.figure(constrained_layout=True, facecolor='0.5')
    fig.set_size_inches([15,7])
    gs = gridspec.GridSpec(num_videos, np.sum(epoch_frame_nums), figure=fig)
    gs.update(wspace=0.0, hspace=0.0)
    for ic, (cam, camname, sync_line) in enumerate(videos_present):
        video_path = paths[cam]
        v = cv2.VideoCapture(video_path)
        frames_of_interest = frames_to_grab[camname]
        
        for i,f in enumerate(frames_of_interest):
            v.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = v.read()
            ax = fig.add_subplot(gs[ic, i])
            ax.imshow(frame)
            #ax.axis('off')
            ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
            
            frame_epoch = get_epoch(i)
            ax.spines['bottom'].set_color(epoch_colors[frame_epoch])
            ax.spines['top'].set_color(epoch_colors[frame_epoch]) 
            ax.spines['right'].set_color(epoch_colors[frame_epoch])
            ax.spines['left'].set_color(epoch_colors[frame_epoch])
    
    for e,ec,name,c in zip(epoch_frame_nums, cumulative_epoch_frames, 
                           ['Behavior', 'Mapping', 'Replay'], epoch_colors):
        
        xcoord = (ec - e/2)/np.sum(epoch_frame_nums)        
        fig.text(xcoord, 0.97, name, color=c, size='large', ha='center')
        
    
    save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix+'video_frames.png'))
    #plt.tight_layout()
    
def get_frames_from_epochs(videos_present, sync, epoch_start_times, epoch_end_times, epoch_frame_nums):
    
    frames = {cam[1]:[] for cam in videos_present}
    for ic, (cam, camname, sync_line) in enumerate(videos_present):
        
        frame_times = sync.get_rising_edges(sync_line, units='seconds')
        for es, ee, enum in zip(epoch_start_times, epoch_end_times, epoch_frame_nums):
            epoch_times = np.linspace(es, ee, enum+2)
            closest_frames = np.searchsorted(frame_times, epoch_times[1:-1])
            frames[camname].extend(closest_frames)
    return frames


def make_metadata_json(behavior_pickle, replay_pickle, FIG_SAVE_DIR, prefix=''):
    
    report = {}
    report['mouse_id'] = replay_pickle['mouse_id']
    report['foraging_id'] = replay_pickle['foraging_id']['value']
    report['start time'] = behavior_pickle['start_time'].strftime('%Y%m%d%H%M%S')
    
    save_file = os.path.join(FIG_SAVE_DIR, prefix+'metadata.json')
    save_json(report, save_file)


def copy_probe_depth_images(paths, FIG_SAVE_DIR, prefix=''):
    
    for file in paths:
        if 'probe_depth' in file:
            source_path = paths[file]
            print(source_path)
            dest_path = os.path.join(FIG_SAVE_DIR, file+'.png')
            if not os.path.exists(os.path.dirname(dest_path)):
                os.mkdir(os.path.dirname(dest_path))
            shutil.copyfile(source_path, dest_path)


def plot_unit_metrics(paths, FIG_SAVE_DIR, prefix=''):
    cols_to_plot = [('presence_ratio', 'fraction of session'),
                    ('isi_viol', 'violation rate'),
                    ('max_drift', 'microns'),
                    ('snr', 'SNR'),
                    ('halfwidth', 'ms'),
                    ('firing_rate', 'Hz'),
                    ('amplitude', 'uV') ]
    metrics_keys = [p for p in paths if 'metrics' in p]
    if len(metrics_keys)>0:
        
        for m in metrics_keys:
            fig, ax = plt.subplots(1, len(cols_to_plot), constrained_layout=True)
            fig.set_size_inches([16, 6])
            fig.suptitle(m + ' unit metrics')
            metrics_file = paths[m]
            if metrics_file is None:
                continue
            metrics_data = pd.read_csv(metrics_file) 
            metrics_data = metrics_data.loc[metrics_data['quality']=='good']
            
            for ic, (col, units) in enumerate(cols_to_plot):
                col_data = metrics_data[col].loc[~np.isinf(metrics_data[col])]
                ax[ic].hist(col_data, bins=20, color='k')
                ax[ic].set_title(col)
                ax[ic].set_xlabel(units)
            
            save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix+m+'_unit_metrics.png'))

def plot_opto_responses(probe_dict, opto_pkl, syncDataset, FIG_SAVE_DIR, prefix='', opto_sample_rate=10000):
    
    opto_stim_table = get_opto_stim_table(syncDataset, opto_pkl, opto_sample_rate=opto_sample_rate)
    levels = np.unique(opto_stim_table['trial_levels'])
    conds = np.unique(opto_stim_table['trial_conditions'])
    
    trial_start_times = opto_stim_table['trial_start_times']
    
    for probe in probe_dict:
        u_df = probe_dict[probe]
        good_units = u_df[(u_df['quality']=='good')&(u_df['snr']>1)]
        spikes = good_units['times']
        peakChans = good_units['peak_channel'].values
        unit_shank_order = np.argsort(peakChans)
        
        fig = plt.figure(constrained_layout=True, facecolor='w')
        fig.set_size_inches([18,10])
        fig.suptitle('Probe {} opto responses'.format(probe))
        gs = gridspec.GridSpec(levels.size*2 + 1, conds.size*10+1, figure=fig)
        #gs = gridspec.GridSpec(levels.size*2 + 1, conds.size, figure=fig)
        color_axes = []
        ims = []
        cond_trial_duration = [0.4, 1.2]
        for ic, cond in enumerate(conds):
            this_waveform = opto_pkl['opto_waveforms'][cond]
            plot_duration = cond_trial_duration[ic]
            ax_wave = fig.add_subplot(gs[0, ic*10:(ic+1)*10])
            ax_wave.plot(np.arange(this_waveform.size)/opto_sample_rate, this_waveform)
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
                psths = np.array([makePSTH_numba(s.flatten(), trial_starts-0.1, plot_duration, 
                                        binSize=0.001, convolution_kernel=0.005, avg=True) for s in spikes])
        
                #bin_times = psths[0, 1, :]
                psths = psths[unit_shank_order, 0, :].squeeze()
                psths_baseline_sub = np.array([p-np.mean(p[:100]) for p in psths])   
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
        
        min_clim_val = -10
        max_clim_val = 20
        
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
        
        save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix+probe+'_optoResponse.png'))


def get_opto_stim_table(syncDataset, opto_pkl, opto_sample_rate=10000):
    
    trial_levels = opto_pkl['opto_levels']
    trial_conds = opto_pkl['opto_conditions']
    trial_start_times = syncDataset.get_rising_edges('stim_trial_opto', units='seconds')
    
    waveforms = opto_pkl['opto_waveforms']
    trial_waveform_durations = [waveforms[cond].size/opto_sample_rate for cond in trial_conds]
    
    trial_end_times = trial_start_times + trial_waveform_durations
    
    trial_dict = {
            'trial_levels': trial_levels,
            'trial_conditions': trial_conds,
            'trial_start_times': trial_start_times,
            'trial_end_times': trial_end_times}
    
    return trial_dict
    
   
def copy_files(file_keys, paths, FIG_SAVE_DIR, prefix=''):
    
    for key in file_keys:
        source_path = paths[key]
        if source_path is not None:
            dest_path = os.path.join(FIG_SAVE_DIR, prefix + os.path.basename(source_path))
            if not os.path.exists(os.path.dirname(dest_path)):
                os.mkdir(os.path.dirname(dest_path))
            
            shutil.copyfile(source_path, dest_path)
        

def save_json(to_save, save_path):
    
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    with open(save_path, 'w') as f:
        json.dump(to_save, f, indent=2)


def save_figure(fig, save_path):
    
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    fig.savefig(save_path)


def save_as_plotly_json(fig, save_path):
    
    plotly_fig = tls.mpl_to_plotly(fig)    
    plotly_fig.write_json(save_path)
    
    
def read_json(path):
    
    with open(path, 'r') as f:
        j = json.load(f)
    
    return j

    
    
