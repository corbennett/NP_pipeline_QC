# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:15:38 2020

@author: svc_ccg
"""
from psycopg2 import connect, extras
import numpy as np
import os, glob, shutil
from visual_behavior.visualization.extended_trials.daily import make_daily_figure
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.ophys.sync import sync_dataset
import pandas as pd
from matplotlib import pyplot as plt
import probeSync, analysis

### SPECIFY LIMS EXPERIMENT TO PULL ####
lims_ecephys_id = '1007083115'


### CONNECT TO LIMS AND GET PATHS TO DATA FOR THIS EXPERIMENT ####
con = connect(
            dbname='lims2',
            user='limsreader',
            host='limsdb2',
            password='limsro',
            port=5432,
        )
con.set_session(
            readonly=True, 
            autocommit=True,
        )
cursor = con.cursor(
            cursor_factory=extras.RealDictCursor,
        )
cursor = con.cursor(
            cursor_factory=extras.RealDictCursor,
        )


QRY = ''' 
    SELECT es.id,
        es.name,
        es.storage_directory,
        es.workflow_state,
        es.date_of_acquisition,
        es.stimulus_name,
        sp.external_specimen_name,
        isi.id AS isi_experiment_id,
        e.name AS rig,
        u.login AS operator,
        p.code AS project,
        bs.storage_directory AS behavior_dir,
        ARRAY_AGG(ep.id ORDER BY ep.id) AS ephys_probe_ids,
        ARRAY_AGG(ep.storage_directory ORDER BY ep.id) AS probe_dirs
    FROM ecephys_sessions es
        JOIN specimens sp ON sp.id = es.specimen_id
        LEFT JOIN isi_experiments isi ON isi.id = es.isi_experiment_id
        LEFT JOIN equipment e ON e.id = es.equipment_id
        LEFT JOIN users u ON u.id = es.operator_id
        JOIN projects p ON p.id = es.project_id
        LEFT JOIN ecephys_probes ep ON ep.ecephys_session_id = es.id
        JOIN behavior_sessions bs ON bs.foraging_id = es.foraging_id
    WHERE es.id = {}
    GROUP BY es.id, sp.external_specimen_name, u.login, e.name, p.code, isi.id, bs.storage_directory
    '''

cursor.execute(QRY.format('1007083115'))
exp_data = cursor.fetchall()[-1]
print(exp_data)

FIG_SAVE_DIR = os.path.join(r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\QC", 
                            exp_data['external_specimen_name']+'_'+exp_data['name'])
if ~os.path.exists(FIG_SAVE_DIR):
    os.mkdir(FIG_SAVE_DIR)
                            

exp_dir = r'\\' + os.path.normpath(exp_data['storage_directory'])[1:]
behavior_dir = r'\\' + os.path.normpath(exp_data['behavior_dir'])[1:]
probe_dirs = ['r\\' + os.path.normpath(pd) for pd in exp_data['probe_dirs']]
sub_dirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir,d)) and not 'eye'in d]
mod_time = np.array([os.path.getmtime(d) for d in sub_dirs])
sub_dirs_ordered = np.array(sub_dirs)[np.argsort(mod_time)]

#FIND MOST RECENT DIRECTORY WITH SYNC and PKL FILES ###
sync_pkl_dir = None
for sd in sub_dirs_ordered[::-1]:
    stim_file = glob.glob(os.path.join(sd, '*stim.pkl'))
    sync_file = glob.glob(os.path.join(sd, '*.sync'))
    
    if len(stim_file)>0 and len(sync_file)>0:
        sync_pkl_dir = sd
        break

#FIND MOST RECENT DIRECTORY WITH SPIKE SORTING RESULTS
spikes_dir = None
for sd in sub_dirs_ordered[::-1]:
    probe_dir = glob.glob(os.path.join(sd, '*probe*'))
    if len(probe_dir)>0 and os.path.isdir(probe_dir[0]):
        spikes_dir = sd
        break


### GET FILE PATHS TO SYNC AND PKL FILES ###
SYNC_FILE = glob.glob(os.path.join(sync_pkl_dir, '*.sync'))[0]
BEHAVIOR_PKL = glob.glob(os.path.join(behavior_dir, '*.pkl'))[0]
REPLAY_PKL = glob.glob(os.path.join(sync_pkl_dir, '*.replay.pkl'))[0]
MAPPING_PKL = glob.glob(os.path.join(sync_pkl_dir, '*.stim.pkl'))[0]

for f,s in zip([SYNC_FILE, BEHAVIOR_PKL, REPLAY_PKL, MAPPING_PKL], ['sync: ', 'behavior: ', 'replay: ', 'mapping: ']):
    print(s + f)


### MAKE DAILY BEHAVIOR PLOT
behavior_data = pd.read_pickle(BEHAVIOR_PKL)
core_data = data_to_change_detection_core(behavior_data)
trials = create_extended_dataframe(
    trials=core_data['trials'],
    metadata=core_data['metadata'],
    licks=core_data['licks'],
    time=core_data['time'])

mapping_data = pd.read_pickle(MAPPING_PKL)
replay_data = pd.read_pickle(REPLAY_PKL)

daily_behavior_fig = make_daily_figure(trials)
daily_behavior_fig.savefig(os.path.join(FIG_SAVE_DIR, 'behavior_summary.png'))


### PLOT FRAME INTERVALS ###
syncDataset = sync_dataset.Dataset(SYNC_FILE)
vr, vf = probeSync.get_sync_line_data(syncDataset, 'stim_vsync')

fig, ax = plt.subplots()
ax.plot(np.diff(vf))
ax.set_ylim([0, 0.2])

behavior_frame_count = len(core_data['time'])
ax.plot(behavior_frame_count, 0.15, 'ko')

expected_break_2 = behavior_frame_count + mapping_data['intervalsms'].size
ax.plot(expected_break_2, 0.15, 'ko')
ax.set_xlabel('frames')
ax.set_ylabel('interval, ms (capped at 0.2)')

fig.savefig(os.path.join(FIG_SAVE_DIR, 'vsync_intervals.png'))

MONITOR_LAG = 0.036
FRAME_APPEAR_TIMES = vf + MONITOR_LAG  

### CHECK THAT NO FRAMES WERE DROPPED FROM SYNC ###
total_pkl_frames = (behavior_data['items']['behavior']['intervalsms'].size +
                    mapping_data['intervalsms'].size +
                    replay_data['intervalsms'].size + 3) #Add 3 since these are all intervals

assert(total_pkl_frames == len(vf))


### GET PROBE INSERTION IMAGE ###
insertion_image = glob.glob(os.path.join(sync_pkl_dir, '*insertionLocation.png'))[0]
shutil.copyfile(insertion_image, os.path.join(FIG_SAVE_DIR, os.path.basename(insertion_image)))

### GET UNIT METRICS AND BUILD UNIT TABLE ###
probe_dirs = [d for d in os.listdir(spikes_dir) if os.path.isdir(os.path.join(spikes_dir, d))]
probe_dict = {a:{} for a in [s[-1] for s in probe_dirs]}

for p in probe_dirs:
    probe = p[-1]
    full_path = os.path.join(spikes_dir, p)
    
    # Get unit metrics for this probe    
    metrics_file = os.path.join(full_path, 'continuous\\Neuropix-PXI-100.0\\metrics.csv')
    unit_metrics = pd.read_csv(metrics_file)
    unit_metrics = unit_metrics.set_index('cluster_id')
    
    # Get unit data
    units = probeSync.getUnitData(full_path, syncDataset)
    units = pd.DataFrame.from_dict(units, orient='index')
    units['cluster_id'] = units.index.astype(int)
    units = units.set_index('cluster_id')
    
    units = pd.merge(unit_metrics, units, left_index=True, right_index=True, how='outer')
    
    probe_dict[probe] = units


### PLOT POPULATION RF FOR EACH PROBE ###
flatten = lambda l: [item[0] for sublist in l for item in sublist]
num_channels_to_take_from_top = 100
for p in probe_dict:
    u_df = probe_dict[p]
    good_units = u_df[(u_df['quality']=='good')&(u_df['snr']>1)]
    max_chan = good_units['peak_channel'].max()
    # take spikes from the top 50 channels as proxy for cortex
    spikes = good_units.loc[good_units['peak_channel']>max_chan-num_channels_to_take_from_top]['times']
    rmats = []
    for s in spikes:
        rmat = analysis.plot_rf(mapping_data, s.flatten(), behavior_frame_count, FRAME_APPEAR_TIMES)
        rmats.append(rmat/rmat.max())
        
#    rmats_normed = np.array([r/r.max() for r in rmats])
    rmats_normed_mean = np.nanmean(rmats, axis=0)
 
    # plot population RF
    fig, ax = plt.subplots()
    title = p + ' population RF'
    fig.suptitle(title)
    ax.imshow(np.mean(rmats_normed_mean, axis=2), origin='lower')
    
    fig.savefig(os.path.join(FIG_SAVE_DIR, title + '.png'))
    



