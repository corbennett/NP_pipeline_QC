# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:18:35 2020

@author: svc_ccg
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

 

def find_spikes_per_trial(spikes, trial_starts, trial_ends):
    tsinds = np.searchsorted(spikes, trial_starts)
    teinds = np.searchsorted(spikes, trial_ends)
    
    return teinds - tsinds
    

def plot_rf(mapping_pkl_data, spikes, first_frame_offset, frameAppearTimes, resp_latency=0.025, plot=True, returnMat=False):
#    if axes is None and plot:
#        fig, ax = plt.subplots()
#        axes = [ax]
    
    rfFlashStimDict = mapping_pkl_data
    rfStimParams = rfFlashStimDict['stimuli'][0] 
    rf_pre_blank_frames = int(rfFlashStimDict['pre_blank_sec']*rfFlashStimDict['fps'])
    first_rf_frame = first_frame_offset + rf_pre_blank_frames
    rf_frameTimes = frameAppearTimes[first_rf_frame:]
    rf_trial_start_times = rf_frameTimes[np.array([f[0] for f in np.array(rfStimParams['sweep_frames'])]).astype(np.int)]
#    monSizePix = rfFlashStimDict['monitor']['sizepix']
#    monHeightCm = monSizePix[1]/monSizePix[0]*rfFlashStimDict['monitor']['widthcm']
#    monDistCm = rfFlashStimDict['monitor']['distancecm']
#    monHeightDeg = np.degrees(2*np.arctan(0.5*monHeightCm/monDistCm))
#    imagePixPerDeg = images[0].shape[0]/monHeightDeg 
#    imageDownsamplePixPerDeg = imagesDownsampled[0].shape[0]/monHeightDeg

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
    
    bestOri = np.unravel_index(np.argmax(respMat), respMat.shape)[-1]

    
#    gridSpacingDeg = xpos[1]-xpos[0]
#    gridSpacingPix = int(round(obj.imageDownsamplePixPerDeg*gridSpacingDeg))
#    r = respMat[:,:,bestOri].copy()
#    r -= r.min()
#    r /= r.max()
#    
#    r = np.repeat(np.repeat(r,gridSpacingPix,axis=0),gridSpacingPix,axis=1)
#    rmap = np.zeros(obj.imagesDownsampled[0].shape)
#    i,j = (int(rmap.shape[s]/2-r.shape[s]/2) for s in (0,1))
#    rmap[i:i+r.shape[0],j:j+r.shape[1]] = r[::-1]
#    rmapColor = plt.cm.magma(rmap)[:,:,:3]
#    rmapColor *= rmap[:,:,None]
#    
#    if plot:
#        for ax,img,imname in zip(axes[:-1],obj.imagesDownsampled,obj.imageNames):
#            img = img.astype(float)
#            img /= 255
#            img *= 1-rmap
#            ax.imshow(rmapColor+img[:,:,None])
#            ax.set_xticks([])
#            ax.set_yticks([])
#            ax.set_ylabel(imname,fontsize=12)
#            
#        axes[-1].imshow(rmap,cmap='magma')
#        axes[-1].set_xticks([])
#        axes[-1].set_yticks([])
#    if returnMat:
    return respMat