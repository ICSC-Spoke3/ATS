"""
Based on the work of Ylenia Maruccia
"""

import os
import FATS
import h5py
import matplotlib
import pandas as pd
import numpy as np

from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
from minisom import MiniSom

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from ..base import AnomalyDetector

# Setup logging
import logging
logger = logging.getLogger(__name__)


class IFSOMAnomalyDetector(AnomalyDetector):
    """
    Uses a Self-Organizing Map (SOM) combined with Isolation Forest (IF) on a set of
    features computed using FATS in order to identify anomalous series within a data set.
    """

    @ staticmethod
    def _compute_features_fats_printid(row, a, sel_col):

        logger.info('Processing for label: %s', row.name)

        # Rimuovo zeri iniziali e finali e ID iniziale
        ##row_trim = np.trim_zeros(row.iloc[1:])
        row_trim = np.trim_zeros(row[sel_col])

        # Costruisco il vettore "time" sulla base della lunghezza della riga_trim
        time_tmp = [i for i in range(len(row_trim))]
        lc_tmp = np.array([row_trim, time_tmp])

        ## Available data as an input:
        ## In case the user does not have all the input vectors mentioned above, 
        ## it is necessary to specify the available data by specifying the list 
        ## of vectors using the parameter Data. In the example below, we calculate
        ## all the features that can be computed with the magnitude and time as an input.
        #a = FATS.FeatureSpace(Data=['magnitude','time'], excludeList=['SlottedA_length','StetsonK_AC'])
        a_calc = a.calculateFeature(lc_tmp)

        return pd.Series(a_calc.result(method='array'), index=a_calc.result(method='features'))

    @staticmethod
    def wide_df_to_timeseries_df(wide_df):
        """
        Input:
            wide_df: index = ID, columns = timestamps
        Output:
            DataFrame with index=timestamps, columns=IDs
        """
        timeseries_df = wide_df.T
        timeseries_df.index.name = 'timestamp'
        return timeseries_df.sort_index()

    @ staticmethod
    def wide_df_to_timeseries_df_with_anomaly(wide_df, anomaly_col="outliers"):
        """
        Input:
            wide_df: index = ID, columns = timestamps + anomaly column
        Output:
            DataFrame with index=timestamps, columns=IDs + anomaly indicator
        """
        # Identify timestamp columns (keep only true timestamps)
        timestamp_cols = pd.to_datetime(
            wide_df.columns, errors='coerce'
        ).notna()

        # Time-series numeric values only
        timeseries_df = wide_df.loc[:, timestamp_cols].T

        # Map anomalies: -1 -> True, 1 -> False
        anomaly = wide_df[anomaly_col].map({-1: True, 1: False})

        # Build a constant anomaly time series for each ID
        anomaly_df = pd.DataFrame(
            {i: [v] * len(timeseries_df) for i, v in anomaly.items()},
            index=timeseries_df.index,
        )
        anomaly_df.columns = [f"{c}_anomaly" for c in anomaly_df.columns]

        timeseries_df = pd.concat([timeseries_df, anomaly_df], axis=1)
        timeseries_df.index.name = "timestamp"

        return timeseries_df.sort_index()

    @staticmethod
    def timeseries_df_to_wide_df(timeseries_df):
        """
        Input:
            timeseries_df: DataFrame with index=timestamp, columns=IDs
        Output:
            DataFrame with index=ID, columns=timestamps
        """
        wide_df = timeseries_df.T
        wide_df.index.name = 'ID'
        wide_df.columns.name = None
        return wide_df.sort_index(axis=1)

    def fit(self, data, som_size_x=9, som_size_y=9, sigma_som=1.0, learning_rate_som=0.5, random_seed_som=29,
            n_iterations_som=1000, neighborhood_function_som='gaussian', n_estimators_if='100', max_samples_if='auto',
            contamination_if=0.05, max_features_if=1, random_state_if=29, *args, **kwargs):
        """
        Fit the model using a Self-Organizing Map (SOM) followed by an Isolation Forest (IF).

        Parameters
        ----------
        data : array-like or pandas.DataFrame
            Input dataset used for training.
        som_size_x : int, default=9
            (SOM parameter) Number of SOM grid nodes along the x-axis.
        som_size_y : int, default=9
            (SOM parameter) Number of SOM grid nodes along the y-axis.
        sigma_som : float, default=1.0
            (SOM parameter) Spread (radius) of the SOM neighborhood function.
        learning_rate_som : float, default=0.5
            (SOM parameter) Learning rate used during SOM training.
        random_seed_som : int, default=29
            (SOM parameter) Random seed used for reproducibility.
        n_iterations_som : int, default=1000
            (SOM parameter) Number of training iterations for the SOM.
        neighborhood_function_som : str, default="gaussian"
            (SOM parameter) Type of neighborhood function to use (e.g., `"gaussian"`).
        n_estimators_if : int or str, default="100"
            (Isolation Forest parameter) Number of base estimators (trees) used in the Isolation Forest.
        max_samples_if : int, float, or {"auto"}, default="auto"
            (Isolation Forest parameter) Number or proportion of samples drawn to train each estimator.
            `"auto"` means using all samples.
        contamination_if : float, default=0.05
            (Isolation Forest parameter) Expected proportion of outliers in the data.
        max_features_if : int or float, default=1
            (Isolation Forest parameter) Number or proportion of features to draw per estimator.
        random_state_if : int, default=29
            (Isolation Forest parameter) Random seed used for reproducibility.
        """

        # Debugging log
        logger.debug("SOM size: ("+ str(som_size_x) + "," + str(som_size_y) + ")")
        logger.debug("SOM sigma = " + str(sigma_som))
        logger.debug("SOM learning rate = " + str(learning_rate_som))
        logger.debug("SOM random seed = " + str(random_seed_som))
        logger.debug("SOM N interations = " + str(n_iterations_som))
        logger.debug('IF n_estimators =' + str(n_estimators_if))
        logger.debug('IF max_samples =' + str(max_samples_if))
        logger.debug('IF contamination =' + str(contamination_if))
        logger.debug('IF max_features =' + str(max_features_if))
        logger.debug('IF random_state =' + str(random_state_if))

        #==============================
        # 0) Prepare data
        #==============================

        if not isinstance(data, pd.DataFrame):
            raise TypeError('Only single Pandas DataFrame are supported')
        df = self.timeseries_df_to_wide_df(data)

        #==============================
        # 1) Compute features (FATS)
        #==============================
        logger.info("Computing FATS features...")
        feature_space = FATS.FeatureSpace(Data=['magnitude','time'], excludeList=['SlottedA_length','StetsonK_AC','StructureFunction_index_21','StructureFunction_index_31','StructureFunction_index_32'])

        # Applico la funzione a ciascuna riga e memorizzo i risultati in una nuova colonna
        df_features = df.apply(self._compute_features_fats_printid, axis=1, args=(feature_space,df.columns))
        logger.info('Done')

        logger.info("Cleaning data")
        # Replace inf with NaN
        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Deleting rows with NaN
        #df_input = df_features.dropna(how = 'any', axis = 0)

        #==============================
        # 2) Train Isolation Forest
        #==============================

        logger.info('Training IF...')
        if_model = IsolationForest(n_estimators = int(n_estimators_if),
                                   max_samples = max_samples_if,
                                   contamination = float(contamination_if),
                                   max_features = float(max_features_if),
                                   random_state = int(max_features_if))
        if_model.fit(df_features)
        logger.info('Done')

        #==============================
        # 3) Train SOM
        #==============================

        # Scaling for training SOM
        logger.info('Scaling dataset for training SOM')
        scaler = StandardScaler()
        df_features_scaled = scaler.fit_transform(df_features)

        # Train the SOM
        logger.info('Init SOM')
        som_size = (int(som_size_x), int(som_size_y))
        som_model = MiniSom(som_size[0],
                              som_size[1],
                              df_features_scaled.shape[1],
                              sigma=float(sigma_som),
                              learning_rate=float(learning_rate_som),
                              random_seed=int(random_seed_som),
                              neighborhood_function=neighborhood_function_som)

        # Init weights
        logger.info('Init SOM weights')
        som_model.random_weights_init(df_features_scaled)

        # Training SOM
        logger.info('Training SOM...')
        som_model.train_random(df_features_scaled, int(n_iterations_som))
        logger.info('Done')

        #==============================
        # 4) Store internally
        #==============================

        self.data = {}
        self.data['data_columns'] = data.columns.tolist()
        self.data['som_model'] = som_model
        self.data['if_model'] = if_model
        self.data['df_features'] = df_features # For inspection purposes
        self.data['df_features_scaled'] = df_features_scaled # For inspection purposes

    def apply(self, data, *args, **kwargs):

        #==============================
        # 0) Prepare data
        #==============================

        if not isinstance(data, pd.DataFrame):
            raise TypeError('Only single Pandas DataFrame are supported')
        df = self.timeseries_df_to_wide_df(data)

        #==============================
        # 1) Compute features (FATS)
        #==============================
        logger.info("Computing FATS features...")
        feature_space = FATS.FeatureSpace(Data=['magnitude','time'], excludeList=['SlottedA_length','StetsonK_AC','StructureFunction_index_21','StructureFunction_index_31','StructureFunction_index_32'])

        # Applico la funzione a ciascuna riga e memorizzo i risultati in una nuova colonna
        df_features = df.apply(self._compute_features_fats_printid, axis=1, args=(feature_space,df.columns))
        logger.info('Done')

        logger.info("Cleaning data")
        # Replace inf with NaN
        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Deleting rows with NaN
        #df_input = df_features.dropna(how = 'any', axis = 0)

        #==============================
        # 2) Apply Isolation Forest
        #==============================

        logger.info('Loading IF...')
        if_model = self.data['if_model']
        logger.info('Done')

        # Scoring
        scores_if = if_model.decision_function(df_features)
        outliers_if = if_model.predict(df_features)

        # Scores normalization and probabilities
        prob_outlier = (scores_if.max() - scores_if) / (scores_if.max() - scores_if.min())
        prob_inlier = 1 - prob_outlier

        #==============================
        # 3) Apply SOM
        #==============================

        # Scaling for training SOM
        logger.info('Scaling dataset for training SOM')
        scaler = StandardScaler()
        df_features_scaled = scaler.fit_transform(df_features)

        # Training SOM
        logger.info('Loading SOM...')
        som_model = self.data['som_model']
        logger.info('Done')

        # Finding Winners
        logger.info('Finding winners')
        w_x, w_y = zip(*[(int(x), int(y)) for x, y in [som_model.winner(d) for d in df_features_scaled]])

        self.data['w_x'] = w_x # For inspection purposes
        self.data['w_y'] = w_y # For inspection purposes


        #==============================
        # 4) Prepare & return results
        #==============================

        results_df = pd.DataFrame(df)
        results_df["winner"] = list(zip(w_x, w_y))
        results_df['scores'] = scores_if
        results_df['outliers'] = outliers_if
        results_df['prob_outlier'] = prob_outlier
        results_df['prob_inlier'] = prob_inlier

        self.data['results_df'] = results_df # For inspection purposes

        # Exmaple
        #          2021-01-01 00:00:00  2021-01-02 00:00:00  2021-01-03 00:00:00  2021-01-04 00:00:00  2021-01-05 00:00:00  ...  winner    scores  outliers  prob_outlier  prob_inlier
        # ID                                                                                                                ...
        # 374107                   0.0                  0.0                  0.0                  0.0                  0.0  ...  (8, 2) -0.005672        -1      1.000000     0.000000
        # 1311700                  0.0                  0.0                  0.0                  0.0                  0.0  ...  (3, 4)  0.032142         1      0.372961     0.627039
        # 508010                   0.0                  0.0                  0.0                  0.0                  0.0  ...  (0, 1)  0.054634         1      0.000000     1.000000
        # 602264                   0.0                  0.0                  0.0                  0.0                  0.0  ...  (0, 7)  0.053066         1      0.025997     0.974003
        #
        # [4 rows x 1100 columns]

        return self.wide_df_to_timeseries_df_with_anomaly(results_df)

    def inspect(self, path=None):
        """
        Generates plots for inspection (or save them in the provided path).

        If only a "fit()" is exceuted:
         1. U-matrix
         2. Activation Map

        If also an "apply()" is called:
         3. Activation Map with Isolation Forest results (pie chart: outliers / non-outliers)
         4. Activation Map colored by IF score and counts of outliers / non-outliers
         5. U-matrix with IF scores overlaid

        """

        #matplotlib.use("Agg")
        logger.info("Creating and saving plots...")

        #==============================
        # 1) U-matrix
        #==============================

        u_matrix = self.data['som_model'].distance_map().T

        # Generate plot
        plt.figure(figsize=(8, 6))
        plt.pcolor(u_matrix, cmap='bone_r')  #plt.imshow
        plt.colorbar()
        plt.title('U-Matrix')

        # Customize ticks
        original_ticks = np.arange(0, u_matrix.shape[0])
        shifted_ticks = original_ticks + 0.5  # Shift by 0.5
        plt.xticks(shifted_ticks, labels=original_ticks)
        plt.yticks(shifted_ticks, labels=original_ticks)

        # Save plot
        plt.tight_layout()
        if path:
            file_name = os.path.dirname(path)+"/umatrix.png"
            plt.savefig(file_name, dpi=300)
            logger.info('Saved {}'.format(file_name))
        else:
            plt.show()
        plt.close()


        #==============================
        # 2) Activation Map
        #==============================

        activation_map = self.data['som_model'].activation_response(self.data['df_features_scaled'])

        # Generate plot
        plt.figure(figsize=(8, 6))
        plt.pcolor(activation_map.T, cmap='Blues')
        plt.colorbar()
        plt.title('Activation Map')

        # Customize ticks
        original_ticks = np.arange(0, activation_map.shape[0])
        shifted_ticks = original_ticks + 0.5  # Shift by 0.5
        plt.xticks(shifted_ticks, labels=original_ticks)
        plt.yticks(shifted_ticks, labels=original_ticks)

        # Save plot
        plt.tight_layout()
        if path:
            file_name = os.path.dirname(path)+"/activation_map.png"
            plt.savefig(file_name, dpi=300)
            logger.info('Saved {}'.format(file_name))
        else:
            plt.show()
        plt.close()


        # Optional, if an "apply" was executed
        if 'results_df' in self.data and 'w_x' in self.data and 'w_y' in self.data:

            #==============================
            # 3) Activation Map & IF res.
            #==============================

            # Initialize the activation map for outliers and non-outliers.
            activation_map_shape = activation_map.shape
            class_counts = defaultdict(lambda: np.zeros(2))  # 2 classi: non-outliers e outliers
            class_counts_prop = defaultdict(lambda: np.zeros(2))

            # Populate the activation map with the count of outliers and non-outliers.
            for i, (x_ind, y_ind) in enumerate(zip(np.array(self.data['w_x']), np.array(self.data['w_y']))):
                if self.data['results_df']['outliers'].iloc[i] == 1:
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
            plt.title('Activation Map & IF res.')

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

            # Save plot
            plt.tight_layout()
            if path:
                file_name = os.path.dirname(path)+"/activation_map_and_if.png"
                plt.savefig(file_name, dpi=300)
                logger.info('Saved {}'.format(file_name))
            else:
                plt.show()
            plt.close()

            #==============================
            # 4) IF score and Outliers
            #==============================

            # Create dictionaries to accumulate the scores and count the objects.
            tot_scores_map = defaultdict(list)  # Punteggi totali
            tot_counts_map = defaultdict(int)   # Conta totali
            inlier_scores_map = defaultdict(list)  # Punteggi degli inliers
            inlier_counts_map = defaultdict(int)   # Conta degli inliers
            outlier_scores_map = defaultdict(list)  # Punteggi degli outliers
            outlier_counts_map = defaultdict(int)   # Conta degli outliers

            # Fill the dictionaries with data.
            for i, (x_ind, y_ind) in enumerate(zip(self.data['w_x'], self.data['w_y'])):
                score = self.data['results_df']['scores'].iloc[i]
                tot_scores_map[(x_ind, y_ind)].append(score)
                tot_counts_map[(x_ind, y_ind)] += 1
                if self.data['results_df']['outliers'].iloc[i] == 1:  # Inliers
                    inlier_scores_map[(x_ind, y_ind)].append(score)
                    inlier_counts_map[(x_ind, y_ind)] += 1
                else:  # Outliers
                    outlier_scores_map[(x_ind, y_ind)].append(score)
                    outlier_counts_map[(x_ind, y_ind)] += 1

            # Create a map for the average scores.
            activation_map_shape = activation_map.shape
            tot_map_mean_scores = np.full(activation_map_shape, np.nan)

            # Compute the mean score for each neuron.
            for (x_ind, y_ind), scores in tot_scores_map.items():
                tot_map_mean_scores[x_ind, y_ind] = np.mean(scores)  # Mean scores

            # Display the map
            plt.figure(figsize=(11, 13.3))
            plt.imshow(tot_map_mean_scores.T, cmap='coolwarm', origin='lower', vmin=-0.1, vmax=0.1) #, origin='lower'
            plt.colorbar(orientation="horizontal", pad=0.05,label='IF score')
            plt.title('IF score and Outliers')

            # Add the outlier count at the center of the cell
            for (i, j), ccc in outlier_counts_map.items():
                count_text = "\n".join([f"Out: {int(ccc)}"])
                plt.text(i , j-0.1 , count_text, ha='center', va='center', color='black')
            for (i, j), ccc in inlier_counts_map.items():
                count_text = "\n".join([f"In: {int(ccc)}"])
                plt.text(i , j+0.1 , count_text, ha='center', va='center', color='black')

            # Save plot
            plt.tight_layout()
            if path:
                file_name=os.path.dirname(path)+"/activ_map_IFscore_outliers.png"
                plt.savefig(file_name, dpi=300)
                logger.info('Saved {}'.format(file_name))
            else:
                plt.show()
            plt.close()

            #==============================
            # 5) U-matrix & IF scores
            #==============================

            # Generate the plot
            plt.figure(figsize=(11, 13.3))

            # Display U-Matrix
            plt.imshow(u_matrix, cmap='bone_r', origin='lower')
            plt.colorbar(orientation="horizontal", pad=0.05, label='Distances')
            plt.title('U-matrix & IF scores')

            # Adding IF scores
            for x_ind in range(self.data['som_model'].get_weights().shape[0]):
                for y_ind in range(self.data['som_model'].get_weights().shape[1]):
                    score = tot_map_mean_scores[(x_ind, y_ind)]
                    if not np.isnan(score):
                        score_text = "\n".join([f"{score:.3f}"])
                        plt.text(x_ind , y_ind , score_text, ha='center', va='center', color='#ca0020')

            # Save plot
            plt.tight_layout()
            plt.tight_layout()
            if path:
                file_name=os.path.dirname(path)+"/umatrix_IFscore.png"
                plt.savefig(file_name, dpi=300)
                logger.info('Saved {}'.format(file_name))
            else:
                plt.show()
            plt.close()

        logger.info("Done.")

