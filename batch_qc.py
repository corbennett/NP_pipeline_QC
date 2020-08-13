# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 19:09:46 2020

@author: svc_ccg
"""

import get_sessions as gs
import os
from run_qc_callable import run_qc

#TODO: LOGGING!!! 

source = r"\\10.128.50.43\sd6.3"
sessions_to_run = gs.get_sessions(source, mouseID='!366122', start_date='20200701')
destination = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\mochi"

for ind, s in enumerate(sessions_to_run):
    
    session_name = os.path.basename(s)
    print('\nRunning QC for session {}, {} in {} \n'
          .format(session_name, ind+1, len(sessions_to_run)))
    
    try:
        run_qc(s, destination)
    
    except Exception as e:
        
        print('Failed to run session {}, due to error {} \n'
              .format(session_name, e))