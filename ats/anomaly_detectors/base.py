# -*- coding: utf-8 -*-
"""Anomaly detectors"""

import pandas as pd
from ats.utils import convert_timeseries_df_to_timeseries, convert_timeseries_to_timeseries_df
from timeseria.datastructures import TimeSeries

# Setup logging
import logging
logger = logging.getLogger(__name__)

class AnomalyDetector():

    def fit(self, data, *args, **kwargs):
        """
        Fit the anomaly detector on some (time series) data.

        Args:
            data (pd.DataFrame or list[pd.DataFrame]): A single time series (as a pandas DataFrame) or a list of time series (each as a pandas DataFrame).
            The index of the each DataFrame must be named "timestamp", and each column should represents a variable. 
        """
        raise NotImplementedError()


    def apply(self, data, *args, **kwargs):
        """
        Apply the anomaly detector on some (time series) data.

        Args:
            data (pd.DataFrame or list[pd.DataFrame]): A single time series (in pandas DataFrame format) or a list of time series (in pandas DataFrame format).
            The index of the data frame(s) must be named "timestamp", and each column is supposed to represents a variable.

        Returns:
            pd.DataFrame or list[pd.DataFrame]: the input data with the "anomaly" flag and optional "anomaly_score" columns added on the time series.
        """
        raise NotImplementedError()

class TimeseriaAnomalyDetector(AnomalyDetector):
    """
    Base class for anomaly detectors wrapped from the timeseria library.
    """  
    
    model_class = None

    def __init__(self, *args, **kwargs):
        if self.model_class is None:
            raise NotImplementedError('Subclasses must define a timeseria model')
        self.model = self.model_class(*args, **kwargs)

    def fit(self, data, *args, **kwargs):
        """
        Fit the timeseria anomaly detector model.
        """      
        if not isinstance(data,pd.DataFrame):
            raise NotImplementedError('Not yet implemented for non DataFrame inputs')
        timeseries_df = data

        # Using timeseria to fit the model
        timeseries = convert_timeseries_df_to_timeseries(timeseries_df)
        model = self.model
        model.fit(timeseries, *args, **kwargs)

    def apply(self, data, *args, **kwargs):
        """
        Apply the timeseria anomaly detector model.
        """     
        if not isinstance(data,pd.DataFrame):
            raise NotImplementedError('Not yet implemented for non DataFrame inputs')
        timeseries_df = data

        # Using timeseria to fit and apply the model
        timeseries = convert_timeseries_df_to_timeseries(timeseries_df)
        timeseries = self.model.apply(timeseries, *args, **kwargs)

        # Convert back to DataFrame
        timeseries_df = convert_timeseries_to_timeseries_df(timeseries)
        timeseries_df['anomaly'] = (timeseries_df['anomaly'].astype(float) != 0).astype(int)

        return timeseries_df