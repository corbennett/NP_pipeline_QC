ECHO off
title Script to get RFs given data directory
ECHO activating environment: RFs
call C:\Users\taminar\AppData\Local\Continuum\anaconda3\Scripts\activate.bat C:\Users\taminar\AppData\Local\Continuum\anaconda3\
call conda activate RFs
ECHO navigating to code directory
call cd C:\Users\taminar\Documents\GitHub\NP_pipeline_QC
set /p data_dir=Base data directory: 
call python get_RFs_standalone.py %data_dir%
cmd \k
