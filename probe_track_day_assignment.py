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
import probe_alignment_data_io as data_io


#TODO: LOGGING!!! 
mouseID = '564012'
ISI_pixels_per_micron = 0.44


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
sessions_to_run = np.array(sessions_to_run)[session_order]



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




# GET INSERTION MOTOR COORDS FROM MOTOR LOC FILES
# look for motor locs that are closest in time to the insertion image
insertion_coords = []
for ind, s in enumerate(sessions_to_run):
    image_file = data_io.glob_file(s, '*surface-image3-left.png')[0]
    insertion_time_stamp = data_io.get_modified_timestamp_from_file(image_file)
    
    motor_locs_file = data_io.glob_file(s, '*motor-locs.csv')[0]
    motor_locs = data_io.read_motor_locs_into_dataframe(motor_locs_file)
    motor_locs = motor_locs.loc[session_dates[ind]]
    
    session_insertion_coords = data_io.find_tap_coordinates(motor_locs,
                                                    data_io.map_newscale_SNs_to_probes(motor_locs),
                                                    insertion_time_stamp)

    insertion_coords.append(session_insertion_coords)

angles = {'A': [-14.21405475, -12.3231693,  -58.84145942],
     'B': [ -17.34136572,  -13.18965862, -118.32166694],
     'C': [ -16.93653005,  -12.27583101, -177.7143598 ],
     'D': [-19.30100945, -16.39715103, 121.32239255],
     'E': [-16.82130366, -13.54745601,  61.47706882],
     'F': [-14.73266944, -13.27092408,   1.81126965]}


def yaw(inputMat, a): # rotation around z axis (heading angle)
    yawMat = np.array([[np.cos(a), -np.sin(a), 0],
               [np.sin(a), np.cos(a), 0],
               [0, 0, 1]])
    return np.dot(inputMat,yawMat)


motor_displacement = {}
reticle_displacement = {}
for p in insertion_coords[0]:
    d1 = np.array(insertion_coords[0][p])
    d2 = np.array(insertion_coords[1][p])
    motor_displacement[p] = d2-d1
    
    p_angles = -np.array(angles[p])*np.pi/180
    
    #reticle_displacement[p] = probe_to_reticle([0,0,0], p_R, d2-d1)
    reticle_displacement[p] = yaw(d2-d1, p_angles[2]+np.pi)*ISI_pixels_per_micron

for ip, p in enumerate('ABCDEF'):
    reticle_d1 = [insertion_points[0][0][ip], insertion_points[0][1][ip]]
    displacement = reticle_displacement[p][:2]
    axes[2].plot([reticle_d1[0], reticle_d1[0]+displacement[0]], [reticle_d1[1], reticle_d1[1] - displacement[1]])

fig, ax = plt.subplots()
for p in reticle_displacement:
    ax.plot([0, reticle_displacement[p][0]], [0, reticle_displacement[p][1]])
ax.legend(['A', 'B', 'C', 'D', 'E', 'F'])
ax.set_aspect('equal')

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