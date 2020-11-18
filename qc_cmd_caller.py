# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 09:30:06 2020

@author: svc_ccg
"""
import os
from run_qc_class import run_qc
import argparse

#TODO: LOGGING!!! 

def call_qc(session, probes_to_run='ABCDEF', cortical_sort=True,
            destination=r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\QC"):

    session_name = os.path.basename(session)
    print('\nRunning QC for session {} \n'
          .format(session_name))
    print('Saving to {}\n'.format(destination))
    
    r=run_qc(session, destination, 
             probes_to_run=probes_to_run, cortical_sort=cortical_sort)
    
    if len(r.errors)>0:
        print('Error(s) encountered: {}  \n'
              .format(r.errors))
    

if __name__ == "__main__":
    
    # run as standalone script
    parser = argparse.ArgumentParser()
    parser.add_argument("session",
                    help="full path to session directory")
    
    parser.add_argument("-p", "--probes", 
                    help= "list of probes to run (default ABCDEF)",
                    default='ABCDEF')
    
    parser.add_argument("-ctx", "--cortical_sort", help="if tag included, run on cortical sort",
                    action="store_true")

    parser.add_argument("-d", "--destination", help="base directory to save QC output",
                    default=r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\QC")
    
    args = parser.parse_args()
    
    call_qc(args.session, args.probes, args.cortical_sort, args.destination)
        
    
    
