#!/usr/bin/env python
# coding: utf-8
#
# FatsFeatures.py
# 
#
# Author: Y. Maruccia
# Creation Date: May 17, 2024
# Modified: YM - October 2025
###############
# USAGE
# 
# python FatsFeatures.py
# 
###############

import pandas as pd
import numpy as np
from scipy import stats
import os
from time import time, localtime, asctime
from utils import check_format, check_file_exists, readFileConfigStats, calcola_features_fats_printid
import FATS

import argparse

parser = argparse.ArgumentParser(description="Run SOM + IF pipeline")
parser.add_argument("config_file", help="Path to the configuration file")
parser.add_argument("log_file", help="Path to the log file")

args = parser.parse_args()

config_file = args.config_file
log_file = args.log_file


filelog=open(log_file,"w")
filelog.write("FatsFeatures.py - Starting Time: "+asctime(localtime(time()) )+"\n\n")
filelog.close()
start=time()

# reading config.ini file
try:
    inputFile, outputFile, inputFormat, outputFormat, id_col = readFileConfigStats(config_file)

except:
    print("\nError reading configuration file! Please Retry")
    filelog=open(log_file,"a")
    filelog.write("Error reading configuration file\n")
    filelog.close()
    exit(1)

# check if input/output exist
check_file_exists(inputFile)
check_file_exists(os.path.dirname(outputFile))

# check if the input/output format is allowed
check_format(inputFormat)
check_format(outputFormat)


filelog=open(log_file,"a")
filelog.write("Input file: "+inputFile+"\n")
filelog.write("Output file: "+outputFile+"\n\n")
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
    column_names_file = "/home/ylenia/BancaIntesaSanPaolo/out/filteredID_2021_2022_2023_col_names.txt"
    with open(column_names_file, 'r') as f:
        column_names = f.read().splitlines()
    df = pd.DataFrame(array_data, columns=column_names)
df['ID'] = df['ID'].astype(int)

read_input_time=time()-start
print("Reading input file duration: %4.2fs\n" % read_input_time)
print("Starting FATS Analysis\n")

filelog=open(log_file,"a")
filelog.write("------ STARTING FATS ANALYSIS  ------\n\n")
filelog.close()


# Available data as an input:
# In case the user does not have all the input vectors mentioned above, 
# it is necessary to specify the available data by specifying the list 
# of vectors using the parameter Data. In the example below, we calculate
# all the features that can be computed with the magnitude and time as an input.
a = FATS.FeatureSpace(Data=['magnitude','time'], excludeList=['SlottedA_length','StetsonK_AC','StructureFunction_index_21','StructureFunction_index_31','StructureFunction_index_32'])

# Escludo l'ID dalle colonne per il calcolo delle features Fats
sel_col = [col for col in df.columns if col != id_col]

# Applico la funzione a ciascuna riga e memorizzo i risultati in una nuova colonna
df_fats = df.apply(calcola_features_fats_printid, axis=1, args=(a,sel_col,id_col))

# Aggiungo la colonna 'ID' al nuovo DataFrame con le features
df_fats.insert(0, id_col, df[id_col])

stats_time=time()-start
print("Fats analysis duration: %4.2fs\n" % stats_time)
print("Writing output...\n")


filelog=open(log_file,"a")
filelog.write("------ WRITING OUTPUT ------\n\n")
filelog.close()


# writing final file
if outputFormat == 'csv':                       # Write DataFrame to csv
    df_fats.to_csv(outputFile, index=False)
elif outputFormat == 'hdf5':                    # Write DataFrame to HDF5
    with h5py.File(outputFile, 'w') as hf:
        hf.create_dataset('df', data=df_fats)
        hf['df'].attrs['column_names'] = df_fats.columns.tolist()
elif outputFormat == 'numpy':                   # Write DataFrame to npy array
    outputFile_wo_ext, ext = os.path.splitext(outputFile)
    column_names_file = outputFile_wo_ext + "_col_names.txt"
    with open(column_names_file, 'w') as f:
        f.write('\n'.join(df_fats.columns.tolist()))
    final_array = df_fats.to_numpy()
    np.save(outputFile, final_array)


output_time=time()-start
print("Writing output duration: %4.2fs\n" % output_time)
print("\nCOMPLETED!")

tempo=time()-start
filelog=open(log_file,"a")
filelog.write("Reading input file duration: %4.2fs\n" % read_input_time)
filelog.write("Fats analysis duration: %4.2fs\n" % stats_time)
filelog.write("Writing output duration: %4.2fs\n" % output_time)
filelog.write("Elapsed time: %4.2fs\n" % tempo)
filelog.close()

print("\n\n------ Elapsed time: %4.2fs\n ------\n\n" % tempo)
