from ..base import AnomalyDetector

import numpy as np
import pandas as pd
from tslies.background.bnnpredictor import BNNPredictor
from tslies.trigger import Trigger
from tslies.utils import Data


class BNNAnomalyDetector(AnomalyDetector):
    def __init__(self, hyperparams='default'):
        """
        Initialize the BNN anomaly detector with the provided hyperparameters.

        Parameters
        ----------
        hyperparams : dict or 'default', optional
            Hyperparameter grid for the BNN model. When set to 'default', a
            predefined configuration is used. The default sets are as follows:
            ```{'units_for_layers' : ([90], [90], [90], [70], [50]),
            'epochs' : [6],
            'bs' : [1000],
            'do' : [0.02],
            'norm' : [0],
            'drop' : [0],
            'opt_name' : ['Adam'],
            'lr' : [0.001],
            'metrics' : ['mae_bnn'], # found in background/losses.py
            'loss_type' : ['negative_log_likelihood_var'] # found in background/losses.py}
            ```

        Examples
        --------
        >>> detector = BNNAnomalyDetector()
        >>> detector.fit(data, source_columns=['sensor1', 'sensor2'], target_columns=['sensor3'])
        >>> results = detector.apply(data, trigger_type='focus', thresholds={'sensor3': 4.5})

        You can also provide custom hyperparameters to fit the BNN model:
        >>> custom_hyperparams = {
        ...     'units_for_layers': ([100], [80], [60]),
        ...     'epochs': [10],
        ...     'bs': [512],
        ...     'do': [0.05],
        ...     'norm': [0],
        ...     'drop': [0.1],
        ...     'opt_name': ['Adam'],
        ...     'lr': [0.001],
        ...     'metrics': ['mae_bnn'],
        ...     'loss_type': ['negative_log_likelihood_var']
        ... }
        >>> detector = BNNAnomalyDetector(hyperparams=custom_hyperparams)
        """
        if hyperparams == 'default':
            self.hyperparams = {
                'units_for_layers' : ([90], [90], [90], [70], [50]),
                'epochs' : [6],
                'bs' : [1000],
                'do' : [0.02],
                'norm' : [0],
                'drop' : [0],
                'opt_name' : ['Adam'],
                'lr' : [0.001],
                'metrics' : ['mae_bnn'], # found in background/losses.py
                'loss_type' : ['negative_log_likelihood_var'] # found in background/losses.py
            }
        else:
            self.hyperparams = hyperparams
        self.nn = None
        self.x_cols = None
        self.y_cols = None

    def fit(self, data, source_columns=None, target_columns=None):
        """
        Train the underlying BNN predictor on the provided time-series data.

        Parameters
        ----------
        data : pandas.DataFrame
            Time-indexed dataframe containing the training samples.
        source_columns : list of str, optional
            Column names used as explanatory variables.
        target_columns : list of str, optional
            Column names representing the target variables.

        Raises
        ------
        ValueError
            If the input data is not a pandas DataFrame or if the required
            column names are not provided.
        """
        data['datetime'] = data.index
        if not isinstance(data, pd.DataFrame):
            raise ValueError('This anomaly detector can work only on a single time series (as a Pandas DataFrames)')
        if source_columns is None or target_columns is None:
            raise ValueError('source_columns and target_columns must be provided')
        self.x_cols = source_columns
        self.y_cols = target_columns
        nn = BNNPredictor(data, self.y_cols, self.x_cols)
        params = nn.get_hyperparams_combinations(self.hyperparams)[0]
        nn.set_hyperparams(params)
        nn.create_model()
        nn.train()
        self.nn = nn

    def apply(self, data, trigger_type='focus', thresholds=None) -> pd.DataFrame:
        """
        Apply the trained BNN model to the provided time-series data and detect anomalies.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Time-indexed dataframe containing the samples to analyze.
        trigger_type : str, optional
            Type of trigger to use for anomaly detection. Default is 'focus'.
        thresholds : dict, optional
            Dictionary specifying the threshold for each target column. If None,
            a default threshold of 5 is used for all target columns.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the detected anomalies.

        Raises
        ------
        ValueError
            If the input data is not a pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError('This anomaly detector can work only on a single time series (as a Pandas DataFrames)')

        if thresholds is None:
            thresholds = {face: 5 for face in self.y_cols}
        
        data['datetime'] = data.index
        self.nn.df_data = data
        _, y_pred_df = self.nn.predict(save_predictions_plot=False)
        df = Data.merge_dfs(data, y_pred_df)

        for face, face_pred in zip(self.y_cols, self.nn.y_pred_cols):
            df[f'{face}_norm'] = (df[face] - df[face_pred]) / df[f'{face}_std']

        trigger = Trigger(df, self.y_cols, self.nn.y_pred_cols, trigger_type=trigger_type, thresholds=thresholds)
        return_df = trigger.run()
        return_df.drop(columns=[col for col in return_df.columns if col.endswith('_std')] + 
                       [col for col in return_df.columns if col.endswith('_norm')] +
                       [col for col in return_df.columns if col.endswith('_pred')], inplace=True)
        return_df.set_index('datetime', inplace=True, drop=True)
        return_df.index.name = 'timestamp'
        return return_df


def generate_timeseries_df(start='2025-06-10 14:00:00',  tz='UTC', freq='h', entries=10, pattern='sin', variables=1):
    if pattern not in ['sin']:
        raise ValueError(f'Unknown pattern "{pattern}"')

    time_index = pd.date_range(
        start=pd.Timestamp(start),
        periods=entries,
        freq=freq,
        tz=tz
    )

    data = {}
    for i in range(variables):
        col_name = 'value' if variables == 1 else f'value_{i+1}'
        data[col_name] = np.sin(np.arange(entries) + i * np.pi / 4)

    num_anomalies = max(1, entries // 20)
    anomaly_indices = np.random.choice(entries, num_anomalies, replace=False)
    for idx in anomaly_indices:
        for i in range(variables):
            col_name = 'value' if variables == 1 else f'value_{i+1}'
            data[col_name][idx] += np.random.uniform(3, 5)

    df = pd.DataFrame(data, index=time_index)
    df.index.name = 'timestamp'
    return df

if __name__ == '__main__':
    timeseries_df = generate_timeseries_df(entries=1000, variables=3)
    print(timeseries_df.head())
    anomaly_detector = BNNAnomalyDetector()
    anomaly_detector.fit(timeseries_df, source_columns=['value_1', 'value_2'], target_columns=['value_3'])
    results_timeseries_df = anomaly_detector.apply(timeseries_df)
    print(results_timeseries_df.head())
    print('Total anomalies detected:', len(results_timeseries_df[results_timeseries_df['anomaly'] == 1]))