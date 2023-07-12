# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:49:08 2023

@author: svc_ccg
"""
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pandas as pd
import glob, os

def get_part_color(part):
    
    if 'cr' in part:
        color = 'g'
    elif 'pupil' in part:
        color = 'r'
    elif 'eye' in part:
        color = 'b'
    
    return color

vbn_vid_files = pd.read_csv(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\vbn_video_paths.csv")

for rind, row in vbn_vid_files.iterrows():
    
    eye_video_file = row['eye_video']
    eye_dlc = row['eye_dlc_output']
    
    eye_fits = pd.read_hdf(eye_dlc).droplevel('scorer', axis='columns')
    eye_parts = eye_fits.columns.get_level_values(0)
    
    session_id = os.path.basename(eye_video_file).split('.')[0]
    save_dir = os.path.join(r'\\allen\programs\mindscope\workgroups\np-exp\VBN_NWB_validation\eye_fits', session_id)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    cap = cv2.VideoCapture(eye_video_file)
    ret = True
    fps = 60
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_plot = np.random.randint(20, 50000, 20)
    for frame_no in frames_to_plot:
        # fno = sf_no / time_length
        cap.set(1, frame_no);
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
    #    if ret:
    #        vid_frames.append(img)
    
        fig, ax = plt.subplots()
        fig.suptitle(frame_no)
        ax.imshow(img)
        frame_coords = eye_fits.iloc[frame_no]
        
        for part in eye_parts:
            
            color = get_part_color(part)
            x = frame_coords[part]['x']
            y = frame_coords[part]['y']
            likelihood = frame_coords[part]['likelihood']
            ax.plot(x, y, color+'+')
            #ax.text(10, 10, likelihood, color='r')
      
        fig.savefig(os.path.join(save_dir, str(frame_no)+'.png'))          
        plt.close('all')
