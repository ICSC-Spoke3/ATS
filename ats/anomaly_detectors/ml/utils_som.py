#!/usr/bin/env python
# coding: utf-8
#
# utils_som.py
# Useful libraries
#
# Author: Y. Maruccia
# Creation Date: August 25, 2025
# 
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
#from scipy import stats
#import FATS
from IPython.utils.io import capture_output

# Load the required parameters from the config.ini file
def loadConfIniSom(): 
    
    global inputFile
    global inputFormat
    global outputFile
    global outFileSom
    global outFileIf
    global training_feat_file
    global id_col
    global som_size_x
    global som_size_y
    global sigma_som
    global learning_rate_som
    global random_seed_som
    global n_iterations_som
    global neighborhood_function_som
    global n_estimators_if
    global max_samples_if
    global contamination_if
    global max_features_if
    global random_state_if
    
    
    
    in_out_section = Config['InOut']
    inputFile = in_out_section['input file']
    inputFormat = in_out_section['input format']
    outputFile = in_out_section['output file csv']
    outFileSom = in_out_section['output file SOM']
    outFileIf = in_out_section['output file IF']
    training_feat_file = in_out_section['training features']
    
    cols_section = Config['ID column name']
    id_col = cols_section['id']
    
    som_parameters_section = Config['SOM parameters']
    som_size_x = som_parameters_section['som_size_x']
    som_size_y = som_parameters_section['som_size_y']
    sigma_som = som_parameters_section['sigma']
    learning_rate_som = som_parameters_section['learning_rate']
    random_seed_som = som_parameters_section['random_seed']
    n_iterations_som = som_parameters_section['n_iterations']
    neighborhood_function_som = som_parameters_section['neighborhood_function']
    
    if_parameters_section = Config['IF parameters']
    n_estimators_if = if_parameters_section['n_estimators']
    max_samples_if = if_parameters_section['max_samples']
    contamination_if = if_parameters_section['contamination']
    max_features_if = if_parameters_section['max_features']
    random_state_if = if_parameters_section['random_state']
    
	
    return inputFile, inputFormat, outputFile, outFileSom, outFileIf, training_feat_file, id_col, som_size_x, som_size_y, sigma_som, learning_rate_som, random_seed_som, n_iterations_som, neighborhood_function_som, n_estimators_if, max_samples_if, contamination_if, max_features_if, random_state_if
    
    
# Read the config.ini file
def readFileConfigSom(fileConf):

    global Config
    
    Config = configparser.ConfigParser()
    Config.read(fileConf)
    inputFile, inputFormat, outputFile, outFileSom, outFileIf, training_feat_file, id_col, som_size_x, som_size_y, sigma_som, learning_rate_som, random_seed_som, n_iterations_som, neighborhood_function_som, n_estimators_if, max_samples_if, contamination_if, max_features_if, random_state_if = loadConfIniSom()
    
    return inputFile, inputFormat, outputFile, outFileSom, outFileIf, training_feat_file, id_col, som_size_x, som_size_y, sigma_som, learning_rate_som, random_seed_som, n_iterations_som, neighborhood_function_som, n_estimators_if, max_samples_if, contamination_if, max_features_if, random_state_if


# Check if files exist
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist! Please, check.")
        filelog=open("log_som.txt","a")
        filelog.write(f"{file_path} does not exist! Please, check.\n")
        exit(1)


#Check file format
def check_format(outputFormat):
    if outputFormat.lower() not in ['csv', 'hdf5', 'numpy']:
        print(f"Format '{outputFormat}' is not valid. Valid options are: csv, hdf5, numpy.")
        exit(1)


# Check if all specified column names are present in the DataFrame
def check_cols_som(df, file_path, column_names):
    missing_columns = [col for col in column_names if col not in df.columns]
    if missing_columns:
        print(f"The following columns are missing in {file_path}: {missing_columns}")
        filelog=open("log_som.txt","a")
        filelog.write(f"The following columns are missing in {file_path}: {missing_columns}\n")
        exit(1)



# Load the required parameters from the config.ini file
def loadConfigFeatToFlag(): 
    
    global inputFile
    global inputFormat
    global inFileSom
    global inFileIf
    global outputFile
    global training_feat_file
    global id_col
    
    in_out_section = Config['InOut']
    inputFile = in_out_section['input file']
    inputFormat = in_out_section['input format']
    inFileSom = in_out_section['input file SOM']
    inFileIf = in_out_section['input file IF']
    outputFile = in_out_section['output file csv']
    training_feat_file = in_out_section['training features']
    
    cols_section = Config['ID column name']
    id_col = cols_section['id']
    
    return inputFile, inputFormat, inFileSom, inFileIf, outputFile, training_feat_file, id_col
    
    
# Read the config.ini file
def readFileConfigFeatToFlag(fileConf):

    global Config
    
    Config = configparser.ConfigParser()
    Config.read(fileConf)
    inputFile, inputFormat, inFileSom, inFileIf, outputFile, training_feat_file, id_col = loadConfigFeatToFlag()
    
    return inputFile, inputFormat, inFileSom, inFileIf, outputFile, training_feat_file, id_col