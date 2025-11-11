#!/usr/bin/env python
# coding: utf-8
#
# FindFlatID.py
# 
#
# Author: Y. Maruccia
# Creation Date: May 21, 2024
# Modified: YM - October 2025
###############
# USAGE
# 
# python FindFlatID.py
# 
###############

import pandas as pd
import numpy as np
import os
from time import time, localtime, asctime
from utils import check_format, check_file_exists, readFileConfigFlat, find_ids_to_remove

import argparse

parser = argparse.ArgumentParser(description="Run SOM + IF pipeline")
parser.add_argument("config_file", help="Path to the configuration file")
parser.add_argument("log_file", help="Path to the log file")

args = parser.parse_args()

config_file = args.config_file
log_file = args.log_file

filelog=open(log_file,"w")
filelog.write("FindFlatID.py - Starting Time: "+asctime(localtime(time()) )+"\n\n")
filelog.close()
start=time()

# reading config.ini file
try:
    inputFile, outputFile_flat, outputFile_fil, inputFormat, outputFormat_flat, outputFormat_fil, id_col = readFileConfigFlat(config_file)

except:
    print("\nError reading configuration file! Please Retry")
    filelog=open(log_file,"a")
    filelog.write("Error reading configuration file\n")
    filelog.close()
    exit(1)

# check if input/output exist
check_file_exists(inputFile)
check_file_exists(os.path.dirname(outputFile_flat))
check_file_exists(os.path.dirname(outputFile_fil))

# check if the input/output format is allowed
check_format(inputFormat)
check_format(outputFormat_flat)
check_format(outputFormat_fil)


filelog=open(log_file,"a")
filelog.write("Input file: "+inputFile+"\n")
filelog.write("Output flat ID file: "+outputFile_flat+"\n\n")
filelog.write("Output filtered file: "+outputFile_fil+"\n\n")
filelog.write("------ READING INPUT FILE ------\n\n")
filelog.close()

# reading input file
if inputFormat == 'csv':                       # Read DataFrame from csv
    df=pd.read_csv(inputFile)
elif inputFormat == 'hdf5':                    # Read DataFrame from HDF5
    # Open the HDF5 file in read mode
    with h5py.File(inputFile, 'r') as hf:
        dataset = hf['df'][:]
        column_names = hf['df'].attrs['column_names'] if 'column_names' in hf['df'].attrs else None
    df = pd.DataFrame(dataset, columns=column_names)
elif inputFormat == 'numpy':                   # Read DataFrame from npy array
    array_data = np.load(inputFile)
    inputFile_wo_ext, ext = os.path.splitext(inputFile)
    column_names_file = inputFile_wo_ext + "_col_names.txt"
    with open(column_names_file, 'r') as f:
        column_names = f.read().splitlines()
    df = pd.DataFrame(array_data, columns=column_names)

read_input_time=time()-start
print("Reading input file duration: %4.2fs\n" % read_input_time)
print("Searching Flat IDs\n")

filelog=open(log_file,"a")
filelog.write("------ Searching Flat IDs ------\n\n")
filelog.close()

# Cerco gli ID con il conto piatto
ids_flat = find_ids_to_remove(df, id_col)

# Creo un dataframe con i dati degli ID con il conto piatto
df_flat = df[df[id_col].isin(ids_flat)]

# Rimuovo gli ID piatti dal dataframe iniziale
df_filtered = df[~df[id_col].isin(ids_flat)]

flat_time=time()-start
print("Flat ID searching time duration: %4.2fs\n" % flat_time)
print("Writing flat ID output: df_flat...\n")


filelog=open(log_file,"a")
filelog.write("------ WRITING OUTPUT ------\n\n")
filelog.close()


# writing flat file
if outputFormat_flat == 'csv':                       # Write DataFrame to csv
    df_flat.to_csv(outputFile_flat, index=False)
elif outputFormat_flat == 'hdf5':                    # Write DataFrame to HDF5
    with h5py.File(outputFile_flat, 'w') as hf:
        hf.create_dataset('df', data=df_flat)
        hf['df'].attrs['column_names'] = df_flat.columns.tolist()
elif outputFormat_flat == 'numpy':                   # Write DataFrame to npy array
    outputFile_wo_ext, ext = os.path.splitext(outputFile_flat)
    column_names_file = outputFile_wo_ext + "_col_names.txt"
    with open(column_names_file, 'w') as f:
        f.write('\n'.join(df_flat.columns.tolist()))
    final_array = df_flat.to_numpy()
    np.save(outputFile_flat, final_array)
    


print("Writing filtered output: df_filtered...\n")

# writing filtered file
if outputFormat_fil == 'csv':                       # Write DataFrame to csv
    df_filtered.to_csv(outputFile_fil, index=False)
elif outputFormat_fil == 'hdf5':                    # Write DataFrame to HDF5
    with h5py.File(outputFile_fil, 'w') as hf:
        hf.create_dataset('df', data=df_filtered)
        hf['df'].attrs['column_names'] = df_filtered.columns.tolist()
elif outputFormat_fil == 'numpy':                   # Write DataFrame to npy array
    outputFile_wo_ext, ext = os.path.splitext(outputFile_fil)
    column_names_file = outputFile_wo_ext + "_col_names.txt"
    with open(column_names_file, 'w') as f:
        f.write('\n'.join(df_filtered.columns.tolist()))
    final_array = df_filtered.to_numpy()
    np.save(outputFile_fil, final_array)


output_time=time()-start
print("Writing output duration: %4.2fs\n" % output_time)
print("\nCOMPLETED!")

tempo=time()-start
filelog=open(log_file,"a")
filelog.write("Reading input file duration: %4.2fs\n" % read_input_time)
filelog.write("Flat ID searching time duration: %4.2fs\n" % flat_time)
filelog.write("Writing output duration: %4.2fs\n" % output_time)
filelog.write("Elapsed time: %4.2fs\n" % tempo)
filelog.close()

print("\n\n------ Elapsed time: %4.2fs\n ------\n\n" % tempo)
