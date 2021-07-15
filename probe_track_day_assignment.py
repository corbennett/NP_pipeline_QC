# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:27:48 2021

@author: svc_ccg
"""
import pandas as pd
import numpy as np
import os
import get_sessions as gs
from matplotlib import pyplot as plt
import glob
import cv2
import query_lims


#TODO: LOGGING!!! 
mouseID = '571520'

# Get D1 and D2 sessions for this mouse
sources = [r"\\10.128.50.43\sd6.3", 
           r"\\10.128.50.20\sd7", r"\\10.128.50.20\sd7.2", 
           r"\\10.128.54.20\sd8", r"\\10.128.54.20\sd8.2"]

sessions_to_run = gs.get_sessions(sources, mouseID=mouseID)
if len(sessions_to_run) != 2:
    print('Found {} sessions'.format(sessions_to_run))
    raise ValueError('ERROR: Found {} sessions for mouse {} instead of expected 2'.format(len(sessions_to_run), mouseID))
    
get_date = lambda x: os.path.basename(x).split('_')[-1][:8]
get_limsID = lambda x: os.path.basename(x).split('_')[0]
session_dates = np.array([get_date(s) for s in sessions_to_run])
session_order = np.argsort(session_dates)

session_dates = session_dates[session_order]
session_limsids = np.array([get_limsID(s) for s in sessions_to_run])[session_order]
sessions_to_run = sessions_to_run[session_order]



# GET ISI TARGET IMAGE FROM LIMS AND DISPLAY INSERTION ANNOTATIONS
SPECIMEN_QRY = '''
    SELECT id FROM specimens WHERE name LIKE '%{}';
    '''

specimen_id = query_lims.query_lims(SPECIMEN_QRY.format(mouseID))[0]['id']

ISI_QRY = '''
            SELECT storage_directory
            FROM isi_experiments
            WHERE specimen_id = {};

'''
isi_storage_dir = query_lims.query_lims(ISI_QRY.format(specimen_id))[0]['storage_directory']
isi_storage_dir = '\\' + os.path.normpath(isi_storage_dir)
target_image_path = glob.glob(os.path.join(isi_storage_dir, '*target_map.tif'))[0]

target_image = cv2.imread(target_image_path)

insertion_points = {}
for ind, s in enumerate(sessions_to_run):
    
    insertion_annotations_file = glob.glob(os.path.join(s, '*area*.csv'))[0]
    insertion_annotations = pd.read_csv(insertion_annotations_file)
    insertion_points[ind] = [insertion_annotations['ISI Pixel Coordinate X'].values,
                             insertion_annotations['ISI Pixel Coordinate Y'].values]



# DISPLAY INSERTION IMAGES 
fig, axes = plt.subplots(1,3)
fig.set_size_inches([18,12])
for ind, s in enumerate(sessions_to_run):
    insertion_im = glob.glob(os.path.join(s, '*fiducial.png'))
    if len(insertion_im)==0:
        raise ValueError('Could not find insertionLocation image in day {} session {}'.format(ind, s))
    
    im = cv2.imread(insertion_im[0])
    axes[ind].imshow(im[:, int(im.shape[1]/2):, ::-1])
    

axes[2].imshow(target_image)
axes[2].plot(insertion_points[0][0], insertion_points[0][1], 'r+', ms=10)
axes[2].plot(insertion_points[1][0], insertion_points[1][1], 'b+', ms=10)
axes[2].legend(['Day 1', 'Day 2'])


   
motor_locs_file = r"\\10.128.54.20\sd8.2\1108528422_571520_20210610\1108528422_571520_20210610.motor-locs.csv"



serialToProbeDict = {' SN32148': 'A', ' SN32142': 'B', ' SN32144':'C', ' SN32149':'D', ' SN32135':'E', ' SN24273':'F'}
serialToProbeDict = {' SN34027': 'A', ' SN31056': 'B', ' SN32141':'C', ' SN32146':'D', ' SN32139':'E', ' SN32145':'F'}

#Date and time of experiment
dateOfInterest = os.path.basename(motor_locs_file).split('_')[-1][:8]
#dateOfInterest = '2020-06-30'
startTime = '0:00'  #I've set it to 12 am, only necessary to change if you did multiple insertions that day
                    #This script just finds the first insertion after the experiment time

fulldf = pd.read_csv(motor_locs_file, header=None, names=['time', 'probeID', 'x', 'y', 'z', 'relx', 'rely', 'relz'])
fulldf['time'] = pd.to_datetime(fulldf['time'])
fulldf = fulldf.set_index('time')

#Find probe trajectories for this experiment
pdf = fulldf.loc[dateOfInterest]