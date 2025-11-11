"""
test_from_feat_to_flag.py  
From features to SOM

Author: Y. Maruccia  
Creation Date: August 28, 2025  

Description:
This script evaluates a trained Self-Organizing Map (SOM) and Isolation Forest (IF) 
model using new feature data. It produces flag outputs, statistics, and visualizations 
to analyze anomalous patterns.  
It represents the **second step** of the workflow and must be executed **after** 
`training_and_save_som.py`.

Usage:
    python test_from_feat_to_flag.py config_feat_to_flag.ini log_feat_to_flag.txt

Dependencies:
- Python 3.10+
- minisom
- h5py
- numpy
- pandas
- matplotlib
- utils_som (functions: readFileConfigFeatToFlag, check_file_exists, check_format, check_cols_som)

Workflow:
1. Run `training_and_save_som.py` to train the SOM and Isolation Forest, 
   saving the trained models and plots.
2. Run `test_from_feat_to_flag.py` to test the trained SOM, generate flags, 
   and produce further visualizations.

Configuration file parameters (`.ini`):

[InOut]
- input file: Path to the input dataset (NumPy array in `.npy` format).
- input format: Format of the input file (e.g., `numpy`).
- input file SOM: Path to the trained SOM model (`.pkl`).
- input file IF: Path to the trained Isolation Forest model (`.pkl`).
- output file csv: Path to the CSV file where test results will be saved.  
  Contains columns: ID, SOM winning neuron, IF score, IF outlier flag (-1: outlier, 1: not-outlier).
- training features: Path to the training features used for consistency check (`.pkl`).

[ID column name]
- id: Name of the identifier column in the input dataset.

Generated plots (saved in the same folder as the output CSV):
1. Activation Map
2. Activation Map with pie charts (outliers vs. non-outliers)
3. Distribution of scores with empirical thresholds
4. Distribution of scores with percentiles
5. Distribution of scores with mean and standard deviation
6. Scatter plots of scores and top 10 anomalous indices
"""


# import libraries
import numpy as np
import pandas as pd
import calendar
import h5py
import os
import sys
from time import time, localtime, asctime

from utils_som import readFileConfigFeatToFlag, check_file_exists, check_format, check_cols_som

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

import pickle

# Use non-interactive backend to avoid opening windows when plotting
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Wedge
from collections import defaultdict

import argparse

parser = argparse.ArgumentParser(description="Run SOM + IF pipeline")
parser.add_argument("config_file", help="Path to the configuration file")
parser.add_argument("log_file", help="Path to the log file")

args = parser.parse_args()

config_file = args.config_file
log_file = args.log_file


with open(log_file,"a") as filelog:
    filelog.write("test_from_feat_to_flag.py - Starting Time: "+asctime(localtime(time()) )+"\n\n")
    filelog.write(f"Using config: {config_file}\n")
    filelog.write("Reading configuration file...\n")
start=time()

# reading config_feat_to_flag.ini file
try:
    inputFile, inputFormat, inFileSom, inFileIf, outputFile, training_feat_file, id_col = readFileConfigFeatToFlag(config_file)

except Exception as e:
    print("\nError reading configuration file! Please retry")
    with open(log_file, "a") as filelog:
        filelog.write(f"Error reading configuration file: {str(e)}\n")
    sys.exit(1)

# check if input/output exist
check_file_exists(inputFile)
print("\nInput file exists.")

# check if the input format is allowed
check_format(inputFormat)
print("\nInput file format is allowed.")

# check if SOM model exists
check_file_exists(inFileSom)
print("\nSOM file exists.")

# check if IF model exists
check_file_exists(inFileIf)
print("\nIF file exists.")

# check if file with training features exists
check_file_exists(training_feat_file)
print("\nFile with training features exists.")

check_file_exists(os.path.dirname(outputFile))
print("\nOutput file dir exists.")



### column_names = [id_col]

# reading input file
if inputFormat == 'csv':                       # Read DataFrame from csv
    df_test_tmp=pd.read_csv(inputFile)
elif inputFormat == 'hdf5':                    # Read DataFrame from HDF5
    # Open the HDF5 file in read mode
    with h5py.File(inputFile, 'r') as hf:
        dataset = hf['df'][:]
        column_names = hf['df'].attrs['column_names'] if 'column_names' in hf['df'].attrs else None
    df_test_tmp = pd.DataFrame(dataset, columns=column_names)
elif inputFormat == 'numpy':                   # Read DataFrame from npy array
    array_data = np.load(inputFile)
    inputFile_wo_ext, ext = os.path.splitext(inputFile)
    column_names_file = inputFile_wo_ext+"_col_names.txt"
    with open(column_names_file, 'r') as f:
        column_names = f.read().splitlines()
    df_test_tmp = pd.DataFrame(array_data, columns=column_names)

# Reading training features from pickle file
with open(training_feat_file, "rb") as f:
    train_cols = pickle.load(f)

# Building df for test
df_test = df_test_tmp[[c for c in df_test_tmp.columns if c in id_col or c in train_cols]]


read_input_time=time()-start
print("Reading input file duration: %4.2fs\n" % read_input_time)
with open(log_file,"a") as filelog:
    filelog.write("Reading input file duration: %4.2fs\n\n" % read_input_time)

# Check if the specified column names are present in the input file
check_cols_som(df_test, inputFile, [id_col])
print("\nCol ID is in the input file.")
with open(log_file,"a") as filelog:
    filelog.write("Col ID is in the input file.\n\n")

print("\nCleaning dataset...")
with open(log_file,"a") as filelog:
    filelog.write("Cleaning dataset...\n\n")
# Replace inf with NaN
df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
# Finding ID to be deleted due to NaN
excluded_ids = df_test[df_test.isna().any(axis=1)][id_col].tolist()
# Deleting rows with NaN
df_test = df_test.dropna(how = 'any', axis = 0)
with open(log_file, "a") as filelog:
    filelog.write("ID deleted in the cleaning phase due to NaN:\n")
    filelog.write(", ".join(map(str, excluded_ids)) + "\n\n")

# Selecting columns for x_test
print("\nSelecting Features for test...")
with open(log_file,"a") as filelog:
    filelog.write("Selecting Features for test...\n\n")
x_test = df_test[[c for c in df_test.columns if c != id_col]]

# Scaling for SOM
print("\nScaling dataset for SOM...")
with open(log_file,"a") as filelog:
    filelog.write("Scaling dataset...\n\n")
scaler = StandardScaler()
x_test_scaled = scaler.fit_transform(x_test)

# Loading SOM and finding winners
print("\nLoading SOM and finding winners...")
with open(log_file,"a") as filelog:
    filelog.write("Loading SOM and finding winners...\n\n")
with open(inFileSom, 'rb') as infile:
  som_trained = pickle.load(infile)
win_x, win_y = zip(*[(int(x), int(y)) for x, y in [som_trained.winner(x) for x in x_test_scaled]])

# Loading IF model and findings inliers/outliers
print("\nLoading IF model and findings inliers/outliers...")
with open(log_file,"a") as filelog:
    filelog.write("Loading IF model and findings inliers/outliers...\n\n")
with open(inFileIf, 'rb') as infile:
  if_model_trained = pickle.load(infile)
scores_if = if_model_trained.decision_function(x_test)
outliers_if = if_model_trained.predict(x_test)


# Creating and saving outputs in file .csv
print("\nCreating and saving file csv...")
with open(log_file,"a") as filelog:
    filelog.write("Creating and saving file csv...\n")
df_tmp = pd.DataFrame(df_test[id_col])
df_tmp["winner"] = list(zip(win_x, win_y))
df_tmp['scores'] = scores_if
df_tmp['outliers'] = outliers_if
df_tmp.to_csv(outputFile, index=False)


# Creating plots
print("\nCreating and saving plots...")
with open(log_file,"a") as filelog:
    filelog.write("Creating and saving plots...\n")

########### 1) Activation Map
file_activation_map_test = os.path.dirname(outputFile)+"/activation_map_test.png"
activation_map_test = som_trained.activation_response(x_test_scaled)

plt.figure(figsize=(8, 6))
plt.pcolor(activation_map_test.T, cmap='Blues')
plt.colorbar()
plt.title('Activation Map Test')
# Customize ticks
original_ticks = np.arange(0, activation_map_test.shape[0])
shifted_ticks = original_ticks + 0.5  # Shift by 0.5
plt.xticks(shifted_ticks, labels=original_ticks)
plt.yticks(shifted_ticks, labels=original_ticks)
#Saving plot
plt.tight_layout()
plt.savefig(file_activation_map_test, dpi=300)
plt.close()




########### 2) Activation Map AND Isolation Forest results
file_activation_map_if_test = os.path.dirname(outputFile)+"/activation_map_and_if_test.png"
#---------- Creating plot ----------
# Initialize the activation map for outliers and non-outliers.
activation_map_shape = activation_map_test.shape
class_counts = defaultdict(lambda: np.zeros(2))  # 2 classi: non-outliers e outliers
class_counts_prop = defaultdict(lambda: np.zeros(2))
# Populate the activation map with the count of outliers and non-outliers.
for i, (x_ind, y_ind) in enumerate(zip(np.array(win_x), np.array(win_y))):
    if df_tmp['outliers'].iloc[i] == 1:
        class_counts[(x_ind, y_ind)][0] += 1  # Non-outliers
    else:
        class_counts[(x_ind, y_ind)][1] += 1  # Outliers
# Calculate the proportions.
for key in class_counts:
    class_counts_prop[key] = class_counts[key] / class_counts[key].sum()
# Define the colors for the pie charts.
colors = ['#1a9641', '#d7191c']  # Green for non-outliers, Red for outliers.
# Plotting here the Activation Map
plt.figure(figsize=(11, 13.3))
plt.pcolor(activation_map_test.T, cmap='Blues')
plt.colorbar(orientation="horizontal", pad=0.05)
# Adding pie charts
for (i, j), prop in class_counts_prop.items():
    total = prop.sum()
    start_angle = 0
    for cls_index, proportion in enumerate(prop):
        if proportion > 0:
            end_angle = start_angle + 360 * proportion
            wedge = Wedge((i + 0.5, j + 0.5), 0.48, start_angle, end_angle, color=colors[cls_index])
            plt.gca().add_patch(wedge)
            start_angle = end_angle
# Add the class count at the center of the cell.
label_names = ['no-out', 'out']
for (i, j), ccc in class_counts.items():
    count_text = "\n".join([f"{label_names[cls_index]}: {int(count)}" for cls_index, count in enumerate(ccc)])
    plt.text(i + 0.5, j + 0.5, count_text, ha='center', va='center', fontsize=11, color='black')
# plt.title('Activation map with IF outliers')
# Customize ticks
original_ticks = np.arange(0, activation_map_test.shape[0])
shifted_ticks = original_ticks + 0.5  # Shift by 0.5
plt.xticks(shifted_ticks, labels=original_ticks)
plt.yticks(shifted_ticks, labels=original_ticks)
#Saving plot
plt.tight_layout()
plt.savefig(file_activation_map_if_test, dpi=300)
plt.close()






########### 3) Distribution of scores with empirical thresholds
file_hist_scores_emp = os.path.dirname(outputFile)+"/hist_scores_empirical_thr_test.png"
plt.hist(scores_if, bins=50, alpha=0.7, color="gray", edgecolor="black")
plt.axvline(0, color="red", linestyle="--", label="Threshold IF")
# empirical thresholds
lower, upper = -0.1, 0.1
#
plt.axvspan(scores_if.min(), lower, color="red", alpha=0.2, label="Anomalous")
plt.axvspan(lower, upper, color="orange", alpha=0.2, label="Borderline")
plt.axvspan(upper, scores_if.max(), color="green", alpha=0.2, label="Normal")
# labels
plt.xlabel("Score")
plt.ylabel("Frequence")
plt.title("Distribution of scores with empirical thresholds")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#Saving plot
plt.tight_layout()
plt.savefig(file_hist_scores_emp, dpi=300)
plt.close()



########### 4) Distribution of scores with percentiles
file_hist_scores_percentiles = os.path.dirname(outputFile)+"/hist_scores_percentiles_test.png"
# Percentiles thresholds: 5 and 95
q_low, q_high = np.percentile(scores_if, [5, 95])
# plot
plt.hist(scores_if, bins=50, alpha=0.7, color="gray", edgecolor="black")
plt.axvline(0, color="red", linestyle="--", label="Threshold IF")
#
plt.axvspan(scores_if.min(), q_low, color="red", alpha=0.2, label="Anomalous")
plt.axvspan(q_low, q_high, color="orange", alpha=0.2, label="Borderline")
plt.axvspan(q_high, scores_if.max(), color="green", alpha=0.2, label="Normal")
# labels
plt.xlabel("Score")
plt.ylabel("Frequence")
plt.title("Distribution of scores with percentiles")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#Saving plot
plt.tight_layout()
plt.savefig(file_hist_scores_percentiles, dpi=300)
plt.close()



########### 5) Distribution of scores with mean and std
file_hist_scores_stats = os.path.dirname(outputFile)+"/hist_scores_stats_test.png"
# Statistical parameters
mean = np.mean(scores_if)
std = np.std(scores_if)
k = 1.
low, high = mean - k*std, mean + k*std
# Hist
plt.hist(scores_if, bins=50, alpha=0.7, color="gray", edgecolor="black")
#
plt.axvline(mean, color="blue", linestyle="--", label="Mean")
plt.axvline(low, color="red", linestyle=":", label=f"mean - {k}*std")
plt.axvline(high, color="green", linestyle=":", label=f"mean + {k}*std")
#
plt.axvspan(scores_if.min(), low, color="red", alpha=0.2, label="Anomalous")
plt.axvspan(low, high, color="orange", alpha=0.2, label="Borderline")
plt.axvspan(high, scores_if.max(), color="green", alpha=0.2, label="Normal")
# labels
plt.xlabel("Score")
plt.ylabel("Frequence")
plt.title(f"Scores with mean ± {k}*std")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#Saving plot
plt.tight_layout()
plt.savefig(file_hist_scores_stats, dpi=300)
plt.close()


########### 5) Scatter plots of scores and top 10 anomalous index
file_scatter_scores_top10 = os.path.dirname(outputFile)+"/scatterplot_scores_top10_anomalous_test.png"
# Percentiles thresholds
q_low, q_high = np.percentile(scores_if, [5, 95])
# Plot scatter
plt.figure(figsize=(10,5))
plt.scatter(range(len(scores_if)), scores_if, c="black", s=20, alpha=0.7, label="Data")
#
plt.axhline(0, color="red", linestyle="--", label="Threshold")
plt.axhline(q_low, color="red", linestyle=":", label="5th percentile")
plt.axhline(q_high, color="green", linestyle=":", label="95th percentile")
#
plt.axhspan(scores_if.min(), q_low, color="red", alpha=0.15, label="Anomalous")
plt.axhspan(q_low, q_high, color="orange", alpha=0.15, label="Borderline")
plt.axhspan(q_high, scores_if.max(), color="green", alpha=0.15, label="Normal")

# Finding top 10 points with lower scores
outlier_idx = np.argsort(scores_if)[:10]
# plotting them in red
plt.scatter(outlier_idx, scores_if[outlier_idx], c="red", s=40, label="Top 10 outliers")
# Labels for the worst
for i in outlier_idx:
    plt.text(i, scores_if[i], str(i), fontsize=8, ha="center", va="bottom", rotation=45)
# labels
plt.xlabel("Index")
plt.ylabel("Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#Saving plot
plt.tight_layout()
plt.savefig(file_scatter_scores_top10, dpi=300)
plt.close()



print("\nEnded!\n"+asctime(localtime(time()) )+"\n\n")
with open(log_file,"a") as filelog:
    filelog.write("ENDED!\n" +asctime(localtime(time()) )+"\n\n")