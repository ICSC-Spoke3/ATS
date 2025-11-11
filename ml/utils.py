#!/usr/bin/env python
# coding: utf-8
#
# utils.py
# Useful libraries
#
# Author: Y. Maruccia
# Creation Date: April 15, 2024
# Modified: YM - October 2025
###############
# USAGE
# 
# 
###############

import configparser
import pandas as pd
import numpy as np
import calendar
import os
from scipy import stats
import FATS
from IPython.utils.io import capture_output

   

# Check if files exist
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist! Please, check.")
        filelog=open("log.txt","a")
        filelog.write(f"{file_path} does not exist! Please, check.\n")
        exit(1)


# Check if all specified column names are present in the DataFrame
def check_cols(df, file_path, column_names, col_amount_prefix):
    missing_columns = [col for col in column_names if col not in df.columns]
    if missing_columns:
        print(f"The following columns are missing in {file_path}: {missing_columns}")
        filelog=open("log.txt","a")
        filelog.write(f"The following columns are missing in {file_path}: {missing_columns}\n")
        exit(1)
    
    matching_columns = [col for col in df.columns if col.startswith(col_amount_prefix)]
    if not matching_columns:
        print(f"No columns starting with '{col_amount_prefix}' found in {file_path}")
        filelog=open("log.txt","a")
        filelog.write(f"No columns starting with '{col_amount_prefix}' found in {file_path}\n")
        exit(1)

#Check file format
def check_format(outputFormat):
    if outputFormat.lower() not in ['csv', 'hdf5', 'numpy']:
        print(f"Format '{outputFormat}' is not valid. Valid options are: csv, hdf5, numpy.")
        exit(1)
        

# Check if all specified column names are present in the DataFrame
def check_colsTab(df, file_path, column_names):
    missing_columns = [col for col in column_names if col not in df.columns]
    if missing_columns:
        print(f"The following columns are missing in {file_path}: {missing_columns}")
        filelog=open("log.txt","a")
        filelog.write(f"The following columns are missing in {file_path}: {missing_columns}\n")
        exit(1)
        
        
##################################
### Libraries for StatisticalAnalysis.py
##################################

# Load the required parameters from the config_stats.ini file
def loadConfIniStats(): 
    
    global inputFile
    global outputFile
    global inputFormat
    global outputFormat
    global id_col
    
    in_out_section = Config['InOut']
    inputFile = in_out_section['input file']
    outputFile = in_out_section['output file']
    inputFormat = in_out_section['input format']
    outputFormat = in_out_section['output format']
    
    cols_section = Config['ID column name']
    id_col = cols_section['id']
	
    return inputFile, outputFile, inputFormat, outputFormat, id_col
    
    
# Read the config_stats.ini file
def readFileConfigStats(fileConf):

    global Config
    
    Config = configparser.ConfigParser()
    Config.read(fileConf)
    inputFile, outputFile, inputFormat, outputFormat, id_col = loadConfIniStats()
    
    return inputFile, outputFile, inputFormat, outputFormat, id_col



##################################
### Libraries for FatsFeatures.py
##################################

#decorator
def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        with capture_output():
            value = func(*args, **kwargs)
        return value
    return func_wrapper


# Definisco una funzione che calcola le feaures FATS
def calcola_features_fats(riga, a):
    # Rimuovo zeri iniziali e finali
    riga_trim = np.trim_zeros(riga)
    # Costruisco il vettore "time" sulla base della lunghezza della riga_trim
    time_tmp = [i for i in range(len(riga_trim))]
    
    lc_tmp = np.array([riga_trim, time_tmp])
    
    ## Available data as an input:
    ## In case the user does not have all the input vectors mentioned above, 
    ## it is necessary to specify the available data by specifying the list 
    ## of vectors using the parameter Data. In the example below, we calculate
    ## all the features that can be computed with the magnitude and time as an input.
    #a = FATS.FeatureSpace(Data=['magnitude','time'], excludeList=['SlottedA_length','StetsonK_AC'])
    a_calc = a.calculateFeature(lc_tmp)
    return pd.Series(a_calc.result(method='array'), index=a_calc.result(method='features'))
    
    
# Definisco una funzione che calcola le feaures FATS
def calcola_features_fats_printid(riga, a, sel_col, id_col):
    #Stampo ID
    ID_tmp = riga[id_col]
    print(f"Processing ID {ID_tmp}")
    # Rimuovo zeri iniziali e finali e ID iniziale
    ##riga_trim = np.trim_zeros(riga.iloc[1:])
    riga_trim = np.trim_zeros(riga[sel_col])
    # Costruisco il vettore "time" sulla base della lunghezza della riga_trim
    time_tmp = [i for i in range(len(riga_trim))]
    
    lc_tmp = np.array([riga_trim, time_tmp])
    
    ## Available data as an input:
    ## In case the user does not have all the input vectors mentioned above, 
    ## it is necessary to specify the available data by specifying the list 
    ## of vectors using the parameter Data. In the example below, we calculate
    ## all the features that can be computed with the magnitude and time as an input.
    #a = FATS.FeatureSpace(Data=['magnitude','time'], excludeList=['SlottedA_length','StetsonK_AC'])
    a_calc = a.calculateFeature(lc_tmp)
    return pd.Series(a_calc.result(method='array'), index=a_calc.result(method='features'))
    
    
    
# Definisco una funzione che calcola le feaures FATS, escludendo i punto con n_days <= 10
@blockPrinting
def calcola_features_fats_sel(riga):
    # Rimuovo zeri iniziali e finali
    riga_trim = np.trim_zeros(riga)
    n_days = len(riga_trim)
    
    if n_days <10:
        a = FATS.FeatureSpace(Data=['magnitude','time'], excludeList=['SlottedA_length','StetsonK_AC'])
        feature_names = a.featureList
        return pd.Series([np.nan] * len(feature_names), index=feature_names)
    else:
        # Costruisco il vettore "time" sulla base della lunghezza della riga_trim
        time_tmp = [i for i in range(n_days)]
        
        lc_tmp = np.array([riga_trim, time_tmp])
        
        # Available data as an input:
        # In case the user does not have all the input vectors mentioned above, 
        # it is necessary to specify the available data by specifying the list 
        # of vectors using the parameter Data. In the example below, we calculate
        # all the features that can be computed with the magnitude and time as an input.
        a = FATS.FeatureSpace(Data=['magnitude','time'], excludeList=['SlottedA_length','StetsonK_AC','StructureFunction_index_21', 'StructureFunction_index_31', 'StructureFunction_index_32'])
        a = a.calculateFeature(lc_tmp)
        return pd.Series(a.result(method='array'), index=a.result(method='features'))
        


##################################
### Libraries for FindFlatID.py
##################################

# Load the required parameters from the config_flat.ini file
def loadConfIniFlat(): 
    
    global inputFile
    global outputFile_flat
    global outputFile_fil
    global inputFormat
    global outputFormat_flat
    global outputFormat_fil
    global id_col
    
    in_out_section = Config['InOut']
    inputFile = in_out_section['input file']
    outputFile_flat = in_out_section['output flat ID file']
    outputFile_fil = in_out_section['output filtered file']
    inputFormat = in_out_section['input format']
    outputFormat_flat = in_out_section['output format flat ID file']
    outputFormat_fil = in_out_section['output format filtered file']
	
    cols_section = Config['ID column name']
    id_col = cols_section['id']
    
    return inputFile, outputFile_flat, outputFile_fil, inputFormat, outputFormat_flat, outputFormat_fil, id_col
    
    
# Read the config_flat.ini file
def readFileConfigFlat(fileConf):

    global Config
    
    Config = configparser.ConfigParser()
    Config.read(fileConf)
    inputFile, outputFile_flat, outputFile_fil, inputFormat, outputFormat_flat, outputFormat_fil, id_col = loadConfIniFlat()
    
    return inputFile, outputFile_flat, outputFile_fil, inputFormat, outputFormat_flat, outputFormat_fil, id_col

        
def find_ids_to_remove(df, id_col):
    ids_to_remove = []

    for index, row in df.iterrows():
        # Rimuovi zeri iniziali e finali
        riga_trim = np.trim_zeros(row.values[1:])  # Escludo la colonna 'ID' usando .values[1:]
        
        ## Se la riga trimmata è vuota, continuo al prossimo ciclo
        #if len(riga_trim) == 0:
        #    continue

        # Calcolo minimo e massimo
        min_val = np.min(riga_trim)
        max_val = np.max(riga_trim)

        # Se minimo e massimo sono uguali, aggiungo l'ID alla lista
        if min_val == max_val:
            ids_to_remove.append(row[id_col])

    return ids_to_remove


# Load the required parameters from the config_shortTS.ini file
def loadConfIniShort(): 
    
    global inputFile
    global outputFile_short
    global outputFile_good
    global inputFormat
    global outputFormat_short
    global outputFormat_good
    global id_col
    
    in_out_section = Config['InOut']
    inputFile = in_out_section['input file']
    outputFile_short = in_out_section['output short ID file']
    outputFile_good = in_out_section['output filtered file']
    inputFormat = in_out_section['input format']
    outputFormat_short = in_out_section['output format short ID file']
    outputFormat_good = in_out_section['output format filtered file']
    
    cols_section = Config['ID column name']
    id_col = cols_section['id']
	
    return inputFile, outputFile_short, outputFile_good, inputFormat, outputFormat_short, outputFormat_good, id_col

# Read the config_shortTS.ini file
def readFileConfigShort(fileConf):

    global Config
    
    Config = configparser.ConfigParser()
    Config.read(fileConf)
    inputFile, outputFile_short, outputFile_good, inputFormat, outputFormat_flat, outputFormat_good, id_col = loadConfIniShort()
    
    return inputFile, outputFile_short, outputFile_good, inputFormat, outputFormat_short, outputFormat_good, id_col
    
    
