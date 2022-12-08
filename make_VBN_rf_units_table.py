# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:56:31 2022

@author: svc_ccg
"""
import os
import pandas as pd
import numpy as np

rf_table_dir = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\supplemental_tables\rfs"
rf_array_dir = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\supplemental_tables\rfs\arrays"

units_table = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\vbn_s3_cache\visual-behavior-neuropixels-0.3.0\project_metadata\units.csv"

units = pd.read_csv(r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\vbn_s3_cache\visual-behavior-neuropixels-0.3.0\project_metadata\units.csv")
units = units.set_index('unit_id')

#channels = pd.read_csv(r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\vbn_s3_cache\visual-behavior-neuropixels-0.3.0\project_metadata\channels.csv")
#channels = channels.set_index('ecephys_channel_id')
#
#unit_channels = units.merge(channels, left_on='ecephys_channel_id', right_index=True)

rf_table_files = [os.path.join(rf_table_dir, f) for f in os.listdir(rf_table_dir) if f.endswith('csv')]
rf_table = pd.read_csv(rf_table_files[0])
all_columns = units.columns.tolist() + rf_table.columns.tolist()

units = units.reindex(columns = all_columns)                

for rffile in rf_table_files:
    
    rfs = pd.read_csv(rffile)
    rfs = rfs.set_index('unit_id')
    units.update(rfs)
    

save_dir = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\supplemental_tables"
units.to_csv(os.path.join(save_dir, 'units_with_rf_stats.csv'))

area_list = ['LGd', 'VISp', 'VISl', 'VISrl', 'LP', 'VISal', 'VISpm', 'VISam']

fig, ax = plt.subplots()
means = []
for a in area_list:
    
    unit_areas = units[(units.p_value_rf<0.01)&units.on_screen_rf&(units.structure_acronym==a)&
                       (units.area_rf<2500)&(units.firing_rate>0.1)&(units.presence_ratio>0.9)]['area_rf']
    means.append(unit_areas.mean())
    h, b = np.histogram(unit_areas, bins = np.arange(0, 2000, 100))
    ax.plot(b[:-1], h/h.sum())

ax.legend(area_list)

area_list = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']
area_arrays = []
big_lgn_array = []
for area in area_list:
    
    area_units = units[(units.p_value_rf<0.001)&units.on_screen_rf&(units.structure_acronym==area)&(units.area_rf>400)]
    print(area, len(area_units))

    for u in area_units.index.values:
        array = np.load(os.path.join(rf_array_dir, str(u)+'.npy'))
        #area_arrays.append(array)
        big_v1_array.append(array)

area_arrays = np.array(area_arrays)
rf_areas = area_units.area_rf.values
rf_area_sort_inds = np.argsort(rf_areas)
array_width = int(np.ceil(16*(len(area_arrays)/144)**0.5))
array_height = int(np.ceil(9*(len(area_arrays)/144)**0.5))
agg_array = np.zeros((array_height*9, array_width*9))
for ia, a in enumerate(area_arrays):
    x = int(ia%array_width)
    y = int(ia/array_width)
    agg_array[9*y:9*y+9, 9*x:9*x+9] = a/a.max()


def stitch(arraylist, rfsize=9):
    array_width = int(np.ceil(16*(len(arraylist)/144)**0.5))
    array_height = int(np.ceil(9*(len(arraylist)/144)**0.5))
    agg_array = np.zeros((array_height*rfsize, array_width*rfsize))
    for ia, a in enumerate(arraylist):
        x = int(ia%array_width)
        y = int(ia/array_width)
        agg_array[rfsize*y:rfsize*y+rfsize, rfsize*x:rfsize*x+rfsize] = a/a.max()
    
    return agg_array


fig, ax = plt.subplots()
ax.imshow(agg_array, cmap='gray')



from numba import njit

@njit
def standard_numba(x):
    
    y = (x - x.mean())/x.std()
    return y.flatten()

@njit
def norm(x):
    return x/x.max()

import cv2
target_image_path = r"C:\Users\svc_ccg\Desktop\paul_allen_pic.png"
target_image_path = r"C:\Users\svc_ccg\Desktop\Presentations\mindscope_allstaff_2022\Tamina_Ramirez.jpg"
target_image_path = r"C:\Users\svc_ccg\Downloads\IMG_1736.JPG"
target_image_path = r"C:\Users\svc_ccg\Downloads\IMG_7001.png"
target_image_path = r"C:\Users\svc_ccg\Downloads\keira-tiff-photos-2022-1009.jpg"
target_image_path = r"C:\Users\svc_ccg\Downloads\Scan 293.jpg"
target_image = cv2.imread(target_image_path)
target_image = np.mean(target_image, axis=2)
#target_image = target_image[250:750, 700:1300]
aspect = target_image.shape[0]/target_image.shape[1]
height =  int(np.round((agg_array.size/aspect)**0.5))
height = height - height%9 + 9
width = int(height*aspect)
width = width - width%9 + 9
target_resample = cv2.resize(target_image, (agg_array.shape[1], agg_array.shape[0]))
target_resample = cv2.resize(target_image, (height, width))
target_arrays = []
for xind in range(len(area_arrays)):
    x = int(xind%array_width)
    y = int(xind/array_width)
    #print(x,y)
    tarray = target_resample[9*y:9*y+9, 9*x:9*x+9]
    target_arrays.append(tarray)

target_arrays = []
for xind in range(len(area_arrays)):
    x = int(xind%(height/9))
    y = int(xind/(height/9))
    #print(x,y)
    tarray = target_resample[9*y:9*y+9, 9*x:9*x+9]
    target_arrays.append(tarray)





target_sorted = np.argsort([norm(a).mean() for a in target_arrays])
rfs_sorted = np.argsort([a.mean() for a in area_arrays])

mosaic_image = np.zeros_like(target_resample)
for rfind, targetind in zip(rfs_sorted, target_sorted):
    
    x = int(targetind%array_width)
    y = int(targetind/array_width)
    #x = int(targetind%(height/9))
    #y = int(targetind/(height/9))
    mosaic_image[9*y:9*y+9, 9*x:9*x+9] = norm(area_arrays[rfind])

plt.figure()
plt.imshow(mosaic_image, cmap='gray')

#np.save(os.path.join(r"C:\Users\svc_ccg\Desktop\Presentations\mindscope_allstaff_2022", 'taminamosaic.npy'), mosaic_image)

cv2.imwrite(os.path.join(r"C:\Users\svc_ccg\Desktop\Presentations\mindscope_allstaff_2022", 'taminamosaic4.png'), 255*mosaic_image)


#Get patch of RFs to zoom in on
aspect = mosaic_image.shape[1]/mosaic_image.shape[0]
patch = mosaic_image[1188:9*5+1188, 99:99+7*9]
new_width = int(np.round(patch.shape[0] * aspect))
embedded_patch = np.zeros((patch.shape[0], new_width))
margins = int(np.round((new_width - patch.shape[1])/2))
embedded_patch[:, margins:margins + patch.shape[1]] = patch
center_tile_alone = np.zeros_like(embedded_patch)
center_tile_alone[2*9:3*9, margins+3*9: margins+4*9] = patch[2*9:3*9, 3*9:4*9]
center_tile = cv2.resize(center_tile_alone, (center_tile_alone.shape[1]*2, center_tile_alone.shape[0]*2), interpolation=3)
center_tile_border = cv2.rectangle(center_tile*255, (2*(margins+27), 2*18), (2*(margins+36), 2*27), color=(180, 180, 180), thickness=1)
center_frames = [center_tile_border]*60
for dimming in np.arange(0, 180, 3):
    dimming=int(dimming)
    center_tile_border_dim = cv2.rectangle(center_tile*255, (2*(margins+27), 2*18), (2*(margins+36), 2*27), 
                                       color=(180-dimming, 180-dimming, 180-dimming), thickness=1)
    center_frames.append(center_tile_border_dim)

embedded_resize = cv2.resize(embedded_patch,(center_tile_alone.shape[1]*2, center_tile_alone.shape[0]*2), interpolation=3)

for fade_in in np.linspace(0, 255, 120):
    fade_in_frame = embedded_resize*fade_in
    fade_in_frame[2*18:3*18, 2*(margins+27):2*(margins+36)] = 255*center_tile[2*18:3*18, 2*(margins+27):2*(margins+36)]
    center_frames.append(fade_in_frame)

center_frames.extend([embedded_resize*255]*180)
np.save(os.path.join(r"C:\Users\svc_ccg\Desktop\Presentations\mindscope_allstaff_2022", 'rf_mosaic_center_patch.npy'), center_frames)

def pad_to_aspect(aspect, image):
    
    image_aspect = image.shape[1]/image.shape[0]
    if image_aspect > aspect:
        new_height = int(np.round(image.shape[1] / aspect))
        margins = int(np.round((new_height - image.shape[0])/2))
        padded = np.zeros((2*margins + image.shape[0], image.shape[1]))
        padded[margins:margins + image.shape[0], :] = image

    else:
        new_width = int(np.round(image.shape[0] * aspect))
        margins = int(np.round((new_width - image.shape[1])/2))
        padded = np.zeros((image.shape[0], 2*margins + image.shape[1]))
    
        padded[:, margins:margins + image.shape[1]] = image
        
    return padded

def zoom(image, start_coords, end_coords, duration):
    
#    startwidth = start_coords[3] - start_coords[1]
#    startheight = start_coords[2] - start_coords[0]
#    
#    finalwidth = end_coords[3] - end_coords[1]
#    finalheight = end_coords[2] - end_coords[0]
    final_aspect = (end_coords[3]-end_coords[1])/(end_coords[2]-end_coords[0])
    start_coords = np.array(start_coords)
    end_coords = np.array(end_coords)
    travel = end_coords - start_coords
    travel_per_frame = np.round(travel/duration).astype(int)
    zoom_frames = []
    for frame in range(duration):
        new_coords = start_coords + travel_per_frame*frame
        new_coords = [max(0, n) for n in new_coords]
        new_coords = [min(e, n) for e, n in zip(list(end_coords[2:])*2, new_coords)]
        print(new_coords)
        frame = image[new_coords[0]:new_coords[2], new_coords[1]:new_coords[3]]
        zoom_frames.append(pad_to_aspect(final_aspect, frame))
    
    zoom_frames.append(image)
    return zoom_frames
    

#### build mosaic up with random squares appearing
mosaic_build_frames = []
mosaic_build = np.zeros_like(mosaic_image)
mosaic_build[1188:9*5+1188, 99:99+7*9] = patch
mosaic_build = cv2.rectangle(mosaic_build*255, (99, 1188), (99+7*9-1, 9*5+1188-1), color=(255, 255, 255), thickness=1)
mosaic_build_frames.extend([mosaic_build.astype(np.uint8)]*120)
for dimming in np.arange(0, 255, 10):
    dimming=int(dimming)
    mosaic_build = cv2.rectangle(mosaic_build, (99, 1188), (99+7*9, 9*5+1188), 
                                 color=(255-dimming, 255-dimming, 255-dimming), thickness=1)

    mosaic_build_frames.append(np.copy(mosaic_build).astype(np.uint8))



build_dur = 5*60
add_per_frame = int(len(rfs_sorted)/build_dur)
target_inds = np.copy(target_sorted)
rf_inds = np.copy(rfs_sorted)
choice_inds = list(np.arange(len(target_inds)))
for ind in range(build_dur):
    theseinds = np.random.choice(choice_inds, add_per_frame, replace=False)
    for addind in theseinds:
        thisind = target_inds[addind]
        x = int(thisind%array_width)
        y = int(thisind/array_width)
        #x = int(targetind%(height/9))
        #y = int(targetind/(height/9))
        rfind = rf_inds[addind]
        mosaic_build[9*y:9*y+9, 9*x:9*x+9] = 255*norm(area_arrays[rfind])
    mosaic_build_frames.append(np.copy(mosaic_build).astype(np.uint8))
    choice_inds = [ci for ci in choice_inds if ci not in theseinds]

mosaic_build_frames.append((mosaic_image*255).astype(np.uint8)) 

mosaic_zoom_out = zoom(mosaic_build, [1188, 99, 9*5+1188, 99+7*9], [0, 0, mosaic_image.shape[0], mosaic_image.shape[1]], 120)


import skvideo
skvideo.setFFmpegPath(r'C:\Users\svc_ccg\Documents\ffmpeg\bin')
import skvideo.io

vid_out_path = r"C:\Users\svc_ccg\Desktop\Presentations\mindscope_allstaff_2022\rf_mosaic_build.mp4"
vid_out = skvideo.io.FFmpegWriter(vid_out_path, inputdict={
          '-r': r'60/1',
        },
        outputdict={
          '-vcodec': 'libx264',
          #'-vcodec': 'libhevc',
          '-pix_fmt': 'gray',
          '-r': r'60/1',
          '-crf': '17',
          #'-max_pixels': '10351',
    })

#for idx, frame in enumerate(full_video):
for frame in center_frames:
    frame = cv2.resize(frame.astype(np.uint8), mosaic_image.shape[::-1], interpolation=3)
    vid_out.writeFrame(frame)  

for frame in mosaic_zoom_out:
    frame = cv2.resize(frame.astype(np.uint8), mosaic_image.shape[::-1], interpolation=3)
    vid_out.writeFrame(frame) 
    
for frame in mosaic_build_frames:
    vid_out.writeFrame(frame) 

vid_out.close()
