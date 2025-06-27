# -*- coding: utf-8 -*-
"""Anomaly detectors"""

import pandas as pd

# Setup logging
import logging
logger = logging.getLogger(__name__)


class AnomalyDetector():

    def apply(self, df):
        raise NotImplementedError()


class MinMaxAnomalyDetector(AnomalyDetector):

    def apply(self, timeseries_df, inplace=False):

        logger.info(f'Applying MinMaxAnomalyDetector with inplace={inplace}')

        if not inplace:
            timeseries_df = timeseries_df.copy()

        for col in timeseries_df.columns:
            anomaly_col = f"{col}_anomaly"
            anomaly_flags = pd.Series(0, index=timeseries_df.index)

            max_idx = timeseries_df[col].idxmax()
            min_idx = timeseries_df[col].idxmin()

            anomaly_flags[max_idx] = 1
            anomaly_flags[min_idx] = 1

            timeseries_df[anomaly_col] = anomaly_flags

        return timeseries_df


