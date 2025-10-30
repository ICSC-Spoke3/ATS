# -*- coding: utf-8 -*-
"""Anomaly detectors"""

import pandas as pd

# Setup logging
import logging
logger = logging.getLogger(__name__)


class AnomalyDetector():

    def fit(self, data, *args, **kwargs):
        """
        Fit the anomaly detector on some (time series) data.

        Args:
            data (pd.DataFrame or set[pd.DataFrame]): A single time series (as a pandas DataFrame) or a set of time series (each as a pandas DataFrame).
            The index of the each DataFrame must be named "timestamp", and each column should represents a variable. 
        """
        raise NotImplementedError()


    def apply(self, data, *args, **kwargs):
        """
        Apply the anomaly detector on some (time series) data.

        Args:
            data (pd.DataFrame or set[pd.DataFrame]): A single time series (in pandas DataFrame format) or a set of time series (in pandas DataFrame format).
            The index of the data frame(s) must be named "timestamp", and each column is supposed to represents a variable.

        Returns:
            pd.DataFrame or set[pd.DataFrame]: the input data with the "anomaly" flag and optional "anomaly_score" columns added.
        """
        raise NotImplementedError()

