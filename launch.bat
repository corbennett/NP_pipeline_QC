ECHO off
title Script to get RFs given data directory
ECHO activating environment: qc
call C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3
call conda activate qc
ECHO navigating to code directory
call cd C:\Users\taminar\Documents\GitHub\NP_pipeline_QC
set /p data_dir=Base data directory: 
call python get_RFs_standalone.py %data_dir%
cmd \k
