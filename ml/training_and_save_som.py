"""
training_and_save_som.py  
From features to SOM

Author: Y. Maruccia  
Creation Date: August 25, 2025  

Description:
This script trains a Self-Organizing Map (SOM) combined with Isolation Forest (IF) 
starting from a set of features, and saves both the trained models and intermediate 
results to file.  
The configuration parameters are provided through a `.ini` file, while process logs 
are written to a text file.  
It represents the **first step** of the workflow and must be executed before 
`test_from_feat_to_flag.py`.

Usage:
    python training_and_save_som.py config_training_som.ini log_training_som.txt

Dependencies:
- Python 3.10+
- minisom
- h5py
- numpy
- pandas
- matplotlib
- utils_som (functions: readFileConfigSom, check_file_exists, check_format, check_cols_som)

Workflow:
1. Run `training_and_save_som.py` to train the SOM and Isolation Forest, 
   saving the trained models and plots.
2. Run `test_from_feat_to_flag.py` to test the trained SOM and generate flags.

Configuration file parameters (`.ini`):

[InOut]
- input file: Path to the input dataset (CSV or other supported format).
- input format: Format of the input file (e.g., `csv`).
- output file csv: Path to the CSV file where the training results will be stored.  
  Contains columns: ID, SOM winning neuron, IF score, IF outlier flag (-1: outlier, 1: not-outlier).
- output file SOM: Path where the trained SOM model (`.pkl`) will be saved.
- output file IF: Path where the trained Isolation Forest model (`.pkl`) will be saved.
- training features: Path to save the features used for training (`.pkl`).

[ID column name]
- id: Name of the identifier column in the input dataset.

[SOM parameters]
- som_size_x: Number of nodes along the x-axis of the SOM grid.
- som_size_y: Number of nodes along the y-axis of the SOM grid.
- sigma: Spread of the neighborhood function.
- learning_rate: Learning rate for SOM training.
- random_seed: Random seed for reproducibility.
- n_iterations: Number of training iterations.
- neighborhood_function: Type of neighborhood function (e.g., `gaussian`).

[IF parameters]
- n_estimators: Number of base estimators in the Isolation Forest.
- max_samples: Number of samples to draw for each estimator (`auto` uses all).
- contamination: Proportion of outliers in the dataset.
- max_features: Number of features to draw per estimator.
- random_state: Random seed for reproducibility.

Generated plots (saved in the same folder as the output CSV):
1. U-matrix
2. Activation Map
3. Activation Map with Isolation Forest results (pie chart: outliers / non-outliers)
4. Activation Map colored by IF score and counts of outliers / non-outliers
5. U-matrix with IF scores overlaid
"""

# import libraries
import numpy as np
import pandas as pd
import calendar
import h5py
import os
import sys
from time import time, localtime, asctime
from utils_som import readFileConfigSom, check_file_exists, check_format, check_cols_som

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from sklearn.ensemble import IsolationForest
from minisom import MiniSom
import pickle

#
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
    filelog.write("training_and_save_som.py - Starting Time: "+asctime(localtime(time()) )+"\n\n")
    filelog.write(f"Using config: {config_file}\n")
    filelog.write("Reading configuration file...\n")
start=time()

# reading config_som.ini file
try:
    inputFile, inputFormat, outputFile, outFileSom, outFileIf, training_feat_file, id_col, som_size_x, som_size_y, sigma_som, learning_rate_som, random_seed_som, n_iterations_som, neighborhood_function_som, n_estimators_if, max_samples_if, contamination_if, max_features_if, random_state_if = readFileConfigSom(config_file)

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

# check if dir for saving SOM file exists
check_file_exists(os.path.dirname(outFileSom))
print("\nOutput file dir exists.")

# print SOM & IF parameters
print('\n\n------------ SOM parameters ------------')
print("SOM size: ("+ str(som_size_x) + "," + str(som_size_y) + ")")
print("sigma = " + str(sigma_som))
print("learning rate = " + str(learning_rate_som))
print("random seed = " + str(random_seed_som))
print("N interations = " + str(n_iterations_som))
print('\n\n------------ IF parameters ------------')
print('n_estimators =', n_estimators_if)
print('max_samples =', max_samples_if)
print('contamination =', contamination_if)
print('max_features =', max_features_if)
print('random_state =', random_state_if)


with open(log_file,"a") as filelog:
    filelog.write("Input file: "+inputFile+"\n")
    filelog.write("Input format: "+inputFormat+"\n\n")
    filelog.write("Col ID: "+id_col+"\n\n")
    filelog.write("\n------ SOM parameters: ------\n")
    filelog.write("som size = ("+str(som_size_x) + "," + str(som_size_y) + ")\n")
    filelog.write("sigma = " + str(sigma_som) + "\n")
    filelog.write("learning rate = " + str(learning_rate_som) + "\n")
    filelog.write("random seed = " + str(random_seed_som) + "\n")
    filelog.write("N interations = " + str(n_iterations_som) + "\n")
    filelog.write("\n------ IF parameters: ------\n")
    filelog.write("n_estimators = " + str(n_estimators_if) + "\n")
    filelog.write("max_samples = " + str(max_samples_if) + "\n")
    filelog.write("contamination = " + str(contamination_if) + "\n")
    filelog.write("max_features = " + str(max_features_if) + "\n")
    filelog.write("random_state = " + str(random_state_if) + "\n")
    filelog.write("\n------ READING INPUT FILE ------\n\n")



###column_names = [id_col]

# reading input file
if inputFormat == 'csv':                       # Read DataFrame from csv
    df_input=pd.read_csv(inputFile)
elif inputFormat == 'hdf5':                    # Read DataFrame from HDF5
    # Open the HDF5 file in read mode
    with h5py.File(inputFile, 'r') as hf:
        dataset = hf['df'][:]
        column_names = hf['df'].attrs['column_names'] if 'column_names' in hf['df'].attrs else None
    df_input = pd.DataFrame(dataset, columns=column_names)
elif inputFormat == 'numpy':                   # Read DataFrame from npy array
    array_data = np.load(inputFile)
    inputFile_wo_ext, ext = os.path.splitext(inputFile)
    column_names_file = inputFile_wo_ext+"_col_names.txt"
    with open(column_names_file, 'r') as f:
        column_names = f.read().splitlines()
    df_input = pd.DataFrame(array_data, columns=column_names)
#df_input[id_col] = df_input[id_col].astype(int)



read_input_time=time()-start
print("Reading input file duration: %4.2fs\n" % read_input_time)
with open(log_file,"a") as filelog:
    filelog.write("Reading input file duration: %4.2fs\n\n" % read_input_time)

# Check if the specified column names are present in the input file
check_cols_som(df_input, inputFile, [id_col])
print("\nCol ID is in the input file.")
with open(log_file,"a") as filelog:
    filelog.write("Col ID is in the input file.\n\n")

print("\nCleaning dataset...")
with open(log_file,"a") as filelog:
    filelog.write("Cleaning dataset...\n\n")
# Replace inf with NaN
df_input.replace([np.inf, -np.inf], np.nan, inplace=True)
# Finding ID to be deleted due to NaN
excluded_ids = df_input[df_input.isna().any(axis=1)][id_col].tolist()
# Deleting rows with NaN
df_input = df_input.dropna(how = 'any', axis = 0)
with open(log_file, "a") as filelog:
    filelog.write("ID deleted in the cleaning phase due to NaN:\n")
    filelog.write(", ".join(map(str, excluded_ids)) + "\n\n")

# Selecting columns for x_train 
print("\nSelecting Features from dataset...")
with open(log_file,"a") as filelog:
    filelog.write("Selecting Features from dataset...\n\n")
x_train = df_input[[c for c in df_input.columns if c != id_col]]

# Save features for future tests
with open(training_feat_file, "wb") as f:
    pickle.dump(x_train.columns.tolist(), f)


# ----------- Training Isolation Forest ----------------
start_training_if_time=time()
print("\nTraining IF...")
with open(log_file,"a") as filelog:
    filelog.write("Training IF...\n\n")
model = IsolationForest(n_estimators = int(n_estimators_if), max_samples = max_samples_if, contamination = float(contamination_if), max_features = float(max_features_if), random_state = int(max_features_if))
model.fit(x_train)

scores_if = model.decision_function(x_train)
outliers_if = model.predict(x_train)

## Scores normalization and probabilities
#prob_outlier = (scores_if.max() - scores_if) / (scores_if.max() - scores_if.min())
#prob_inlier = 1 - prob_outlier


training_if_time=time()-start_training_if_time
print("\nTraining IF duration: %4.2fs\n" % training_if_time)
with open(log_file,"a") as filelog:
    filelog.write("Training IF duration: %4.2fs\n\n" % training_if_time)
# ----------- END Training IF --------------



# Scaling for training SOM
print("\nScaling dataset for training SOM...")
with open(log_file,"a") as filelog:
    filelog.write("Scaling dataset...\n\n")
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train)



# ----------- Training SOM --------------
start_training_som_time=time()
print("\nTraining SOM...")
with open(log_file,"a") as filelog:
    filelog.write("Training SOM...\n\n")
som_size = (int(som_size_x), int(som_size_y)) 
som_trained = MiniSom(som_size[0], 
                      som_size[1], 
                      x_scaled.shape[1], 
                      sigma=float(sigma_som), 
                      learning_rate=float(learning_rate_som), 
                      random_seed=int(random_seed_som), 
                      neighborhood_function=neighborhood_function_som)

# Init weights
print("\n1) weights")
with open(log_file,"a") as filelog:
    filelog.write("1) weights\n")
som_trained.random_weights_init(x_scaled)

# Training SOM
print("\n2) training")
with open(log_file,"a") as filelog:
    filelog.write("2) training\n")
som_trained.train_random(x_scaled, int(n_iterations_som))

# Finding Winners
print("\n3) finding winners")
with open(log_file,"a") as filelog:
    filelog.write("3) finding winners\n")
w_x, w_y = zip(*[(int(x), int(y)) for x, y in [som_trained.winner(d) for d in x_scaled]])
# ----------- END Training SOM --------------

training_som_time=time()-start_training_som_time
print("\nTraining SOM duration: %4.2fs\n" % training_som_time)
with open(log_file,"a") as filelog:
    filelog.write("Training SOM duration: %4.2fs\n\n" % training_som_time)
    filelog.write("Saving SOM\n")


# saving the SOM model in the file som.p
print("\nSaving SOM model...")
start_saving_som_time=time()
with open(outFileSom, 'wb') as outfile:
    pickle.dump(som_trained, outfile)

saving_som_time=time()-start_saving_som_time
print("\nSaving SOM model duration: %4.2fs\n" % saving_som_time)
with open(log_file,"a") as filelog:
    filelog.write("Saving SOM model duration: %4.2fs\n\n" % saving_som_time)


# saving the IF model in the file som.p
print("\nSaving IF model...")
start_saving_if_time=time()
with open(outFileIf, 'wb') as outfile:
    pickle.dump(model, outfile)

saving_if_time=time()-start_saving_if_time
print("\nSaving IF model duration: %4.2fs\n" % saving_if_time)
with open(log_file,"a") as filelog:
    filelog.write("Saving IF model duration: %4.2fs\n\n" % saving_if_time)


# Creating and saving outputs in file .csv
print("\nCreating and saving file csv...")
with open(log_file,"a") as filelog:
    filelog.write("Creating and saving file csv...\n") 
df_if_som = pd.DataFrame(df_input[id_col])
df_if_som["winner"] = list(zip(w_x, w_y))
df_if_som['scores'] = scores_if
df_if_som['outliers'] = outliers_if
##df_if_som['prob_outlier'] = prob_outlier
##df_if_som['prob_inlier'] = prob_inlier
df_if_som.to_csv(outputFile, index=False)


# Creating plots
print("\nCreating and saving plots...")
with open(log_file,"a") as filelog:
    filelog.write("Creating and saving plots...\n") 
    
########### 1) U-matrix
file_umatrix = os.path.dirname(outFileSom)+"/umatrix.png"
u_matrix = som_trained.distance_map().T

#---------- Creating plot ----------
plt.figure(figsize=(8, 6))
plt.pcolor(u_matrix, cmap='bone_r')  #plt.imshow
plt.colorbar()
plt.title('U-Matrix')
# Customize ticks
original_ticks = np.arange(0, u_matrix.shape[0])
shifted_ticks = original_ticks + 0.5  # Shift by 0.5
plt.xticks(shifted_ticks, labels=original_ticks)
plt.yticks(shifted_ticks, labels=original_ticks)
#Saving plot
plt.tight_layout()
plt.savefig(file_umatrix, dpi=300)
plt.close()
#---------- END Creating plot ----------


########### 2) Activation Map
file_activation_map = os.path.dirname(outFileSom)+"/activation_map.png"
activation_map = som_trained.activation_response(x_scaled)

#---------- Creating plot ----------
plt.figure(figsize=(8, 6))
plt.pcolor(activation_map.T, cmap='Blues')
plt.colorbar()
plt.title('Activation Map')
# Customize ticks
original_ticks = np.arange(0, activation_map.shape[0])
shifted_ticks = original_ticks + 0.5  # Shift by 0.5
plt.xticks(shifted_ticks, labels=original_ticks)
plt.yticks(shifted_ticks, labels=original_ticks)
#Saving plot
plt.tight_layout()
plt.savefig(file_activation_map, dpi=300)
plt.close()
#---------- END Creating plot ----------


########### 3) Activation Map AND Isolation Forest results
file_activation_map_if = os.path.dirname(outFileSom)+"/activation_map_and_if.png"


#---------- Creating plot ----------
# Initialize the activation map for outliers and non-outliers.
activation_map_shape = activation_map.shape
class_counts = defaultdict(lambda: np.zeros(2))  # 2 classi: non-outliers e outliers
class_counts_prop = defaultdict(lambda: np.zeros(2))
# Populate the activation map with the count of outliers and non-outliers.
for i, (x_ind, y_ind) in enumerate(zip(np.array(w_x), np.array(w_y))):
    if df_if_som['outliers'].iloc[i] == 1:
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
plt.pcolor(activation_map.T, cmap='Blues')
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
original_ticks = np.arange(0, activation_map.shape[0])
shifted_ticks = original_ticks + 0.5  # Shift by 0.5
plt.xticks(shifted_ticks, labels=original_ticks)
plt.yticks(shifted_ticks, labels=original_ticks)
#Saving plot
plt.tight_layout()
plt.savefig(file_activation_map_if, dpi=300)
plt.close()


########### 4) IF score and Outliers
file_activation_map_if_score_out = os.path.dirname(outFileSom)+"/activ_map_IFscore_outliers.png"


#---------- Creating plot ----------
# Step 1: Create dictionaries to accumulate the scores and count the objects.
tot_scores_map = defaultdict(list)  # Punteggi totali
tot_counts_map = defaultdict(int)   # Conta totali
inlier_scores_map = defaultdict(list)  # Punteggi degli inliers
inlier_counts_map = defaultdict(int)   # Conta degli inliers
outlier_scores_map = defaultdict(list)  # Punteggi degli outliers
outlier_counts_map = defaultdict(int)   # Conta degli outliers
# Fill the dictionaries with data.
for i, (x_ind, y_ind) in enumerate(zip(w_x, w_y)):
    score = df_if_som['scores'].iloc[i]
    tot_scores_map[(x_ind, y_ind)].append(score)
    tot_counts_map[(x_ind, y_ind)] += 1
    if df_if_som['outliers'].iloc[i] == 1:  # Inliers
        inlier_scores_map[(x_ind, y_ind)].append(score)
        inlier_counts_map[(x_ind, y_ind)] += 1
    else:  # Outliers
        outlier_scores_map[(x_ind, y_ind)].append(score)
        outlier_counts_map[(x_ind, y_ind)] += 1
# Step 2: Create a map for the average scores.
activation_map_shape = activation_map.shape
tot_map_mean_scores = np.full(activation_map_shape, np.nan)
# Compute the mean score for each neuron.
for (x_ind, y_ind), scores in tot_scores_map.items():
    tot_map_mean_scores[x_ind, y_ind] = np.mean(scores)  # Mean scores
# Step 3: Display the map.
plt.figure(figsize=(11, 13.3))
plt.imshow(tot_map_mean_scores.T, cmap='coolwarm', origin='lower', vmin=-0.1, vmax=0.1) #, origin='lower'
plt.colorbar(orientation="horizontal", pad=0.05,label='IF score')
# Step 4: Add the outlier count at the center of the cell
for (i, j), ccc in outlier_counts_map.items():
    count_text = "\n".join([f"Out: {int(ccc)}"])
    plt.text(i , j-0.1 , count_text, ha='center', va='center', color='black')
for (i, j), ccc in inlier_counts_map.items():
    count_text = "\n".join([f"In: {int(ccc)}"])
    plt.text(i , j+0.1 , count_text, ha='center', va='center', color='black')
#Saving plot
plt.tight_layout()
plt.savefig(file_activation_map_if_score_out, dpi=300)
plt.close()



########### 5) U-matrix and IF scores
file_umatrix_if_score = os.path.dirname(outFileSom)+"/umatrix_IFscore.png"


#---------- Creating plot ----------
plt.figure(figsize=(11, 13.3))
# Display U-Matrix
plt.imshow(u_matrix, cmap='bone_r', origin='lower')
plt.colorbar(orientation="horizontal", pad=0.05, label='Distances')
# Adding IF scores
for x_ind in range(som_trained.get_weights().shape[0]):
    for y_ind in range(som_trained.get_weights().shape[1]):
        score = tot_map_mean_scores[(x_ind, y_ind)]
        if not np.isnan(score):
            score_text = "\n".join([f"{score:.3f}"])
            plt.text(x_ind , y_ind , score_text, ha='center', va='center', color='#ca0020')

#Saving plot
plt.tight_layout()
plt.savefig(file_umatrix_if_score, dpi=300)
plt.close()


print("\nEnded!\n"+asctime(localtime(time()) )+"\n\n")
with open(log_file,"a") as filelog:
    filelog.write("ENDED!\n" +asctime(localtime(time()) )+"\n\n")