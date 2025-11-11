#!/usr/bin/env python
# coding: utf-8
#
# findShortSeries.py
# 
#
# Author: Y. Maruccia
# Creation Date: May 23, 2024
# Modified: YM - October, 2025
###############
# USAGE
# 
# python findShortSeries.py
# 
###############

import numpy as np
import pandas as pd
import os
from time import time, localtime, asctime
from utils import check_format, check_file_exists, readFileConfigShort

len_days = 30

import argparse

parser = argparse.ArgumentParser(description="Run SOM + IF pipeline")
parser.add_argument("config_file", help="Path to the configuration file")
parser.add_argument("log_file", help="Path to the log file")

args = parser.parse_args()

config_file = args.config_file
log_file = args.log_file

filelog=open(log_file,"w")
filelog.write("findShortSeries.py - Starting Time: "+asctime(localtime(time()) )+"\n\n")
filelog.close()
start=time()



# reading config.ini file
try:
    inputFile, outputFile_short, outputFile_good, inputFormat, outputFormat_short, outputFormat_good, id_col = readFileConfigShort(config_file)

except:
    print("\nError reading configuration file! Please Retry")
    filelog=open(log_file,"a")
    filelog.write("Error reading configuration file\n")
    filelog.close()
    exit(1)

# check if input/output exist
check_file_exists(inputFile)
check_file_exists(os.path.dirname(outputFile_short))
check_file_exists(os.path.dirname(outputFile_good))

# check if the input/output format is allowed
check_format(inputFormat)
check_format(outputFormat_short)
check_format(outputFormat_good)


filelog=open(log_file,"a")
filelog.write("Input file: "+inputFile+"\n")
filelog.write("Output short ID file: "+outputFile_short+"\n\n")
filelog.write("Output filtered file: "+outputFile_good+"\n\n")
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
    
df = df.astype(dtype={id_col: int})

read_input_time=time()-start
print("Reading input file duration: %4.2fs\n" % read_input_time)
print("Searching short series IDs\n")

filelog=open(log_file,"a")
filelog.write("------ Searching short series IDs ------\n\n")
filelog.close()


# Escludo l'ID dalle colonne per il calcolo delle features Fats
sel_col = [col for col in df.columns if col != id_col]

# Creo una maschera booleana per i valori diversi da zero
non_zero_mask = df[sel_col].ne(0)

# Trovo il primo indice diverso da zero per ogni riga
first_non_zero_index = non_zero_mask.idxmax(axis=1)

# Trovo l'ultimo indice diverso da zero per ogni riga
last_non_zero_index = non_zero_mask.iloc[:, ::-1].idxmax(axis=1)

# Converto gli indici di colonne in posizioni numeriche
first_non_zero_position = non_zero_mask.values.argmax(axis=1)

# Per l'ultimo indice, calcolo le posizioni numeriche considerando l'inversione
last_non_zero_position = df[sel_col].shape[1] - non_zero_mask.iloc[:, ::-1].values.argmax(axis=1) - 1

# Calcolo il conteggio degli elementi diversi da zero per riga
non_zero_count = non_zero_mask.sum(axis=1)

# Creo il dataframe con le informazioni
df_info = pd.DataFrame({
    'ID': df[id_col],
    'n_days': non_zero_count,
    'first_day': first_non_zero_position,
    'last_day': last_non_zero_position
    })

# Cerco gli ID con conto < 30 days
id_short = df_info[df_info.n_days < len_days][id_col].to_list()

# Creo un dataframe con i dati degli ID con il conto piatto
df_short = df[df[id_col].isin(id_short)]

# Rimuovo gli ID piatti dal dataframe iniziale
df_good = df[~df[id_col].isin(id_short)]


short_time=time()-start
print("Short series ID searching time duration: %4.2fs\n" % short_time)
print("Writing short series ID output: df_short...\n")


filelog=open(log_file,"a")
filelog.write("------ WRITING OUTPUT ------\n\n")
filelog.close()



# writing short file
if outputFormat_short == 'csv':                       # Write DataFrame to csv
    df_short.to_csv(outputFile_short, index=False)
elif outputFormat_short == 'hdf5':                    # Write DataFrame to HDF5
    with h5py.File(outputFile_short, 'w') as hf:
        hf.create_dataset('df', data=df_short)
        hf['df'].attrs['column_names'] = df_short.columns.tolist()
elif outputFormat_short == 'numpy':                   # Write DataFrame to npy array
    outputFile_short_wo_ext, ext = os.path.splitext(outputFile_short)
    column_names_file = outputFile_short_wo_ext + "_col_names.txt"
    with open(column_names_file, 'w') as f:
        f.write('\n'.join(df_short.columns.tolist()))
    final_array = df_short.to_numpy()
    np.save(outputFile_short, final_array)
    


print("Writing filtered output: df_good...\n")

# writing filtered file
if outputFormat_good == 'csv':                       # Write DataFrame to csv
    df_good.to_csv(outputFile_good, index=False)
elif outputFormat_good == 'hdf5':                    # Write DataFrame to HDF5
    with h5py.File(outputFile_good, 'w') as hf:
        hf.create_dataset('df', data=df_good)
        hf['df'].attrs['column_names'] = df_good.columns.tolist()
elif outputFormat_good == 'numpy':                   # Write DataFrame to npy array
    outputFile_good_wo_ext, ext = os.path.splitext(outputFile_good)
    column_names_file = outputFile_good_wo_ext + "_col_names.txt"
    with open(column_names_file, 'w') as f:
        f.write('\n'.join(df_good.columns.tolist()))
    final_array = df_good.to_numpy()
    np.save(outputFile_good, final_array)
    


outputFile_short_wo_ext, ext = os.path.splitext(outputFile_short)
outputFile_short_info = outputFile_short_wo_ext + "_info.csv"

df_info[df_info.n_days < len_days].to_csv(outputFile_short_info, index=False)


outputFile_good_wo_ext, ext = os.path.splitext(outputFile_good)
outputFile_filtered_info = outputFile_good_wo_ext + "_info.csv"

df_info[~df_info.n_days < 30].to_csv(outputFile_filtered_info, index=False)




output_time=time()-start
print("Writing output duration: %4.2fs\n" % output_time)
print("\nCOMPLETED!")

tempo=time()-start
filelog=open("log_flat.txt","a")
filelog.write("Reading input file duration: %4.2fs\n" % read_input_time)
filelog.write("Flat ID searching time duration: %4.2fs\n" % short_time)
filelog.write("Writing output duration: %4.2fs\n" % output_time)
filelog.write("Elapsed time: %4.2fs\n" % tempo)
filelog.close()

print("\n\n------ Elapsed time: %4.2fs\n ------\n\n" % tempo)





