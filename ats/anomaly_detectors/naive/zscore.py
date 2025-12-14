# -*- coding: utf-8 -*-
"""Anomaly detectors"""

import pandas as pd

from ..base import AnomalyDetector

# Setup logging
import logging
logger = logging.getLogger(__name__)


class ZScoreAnomalyDetector(AnomalyDetector):
    
    @AnomalyDetector.apply_method
    def apply(self, data, inplace=False):

        logger.info(f'Applying ZScoreAnomalyDetector with inplace={inplace}')

        if not isinstance(data, pd.DataFrame):
            raise NotImplementedError('This anomaly detector can work only on a single time series (as a Pandas DataFrames)')

        timeseries_df = data

        if not inplace:
            timeseries_df = timeseries_df.copy()

        for col in timeseries_df.columns:
            anomaly_col = f"{col}_anomaly"
            anomaly_flags = pd.Series(0, index=timeseries_df.index)

            mean = timeseries_df[col].mean()
            std = timeseries_df[col].std()
            z = (timeseries_df[col] - mean) / std
            # Check for all-NaN or empty z before calling idxmax/idxmin
            if z.dropna().empty:
                logger.warning(f"Column '{col}' has all NaN or constant values; skipping anomaly detection for this column.")
                timeseries_df[anomaly_col] = anomaly_flags
                continue
            # Check for empty or all-NaN z-score Series before calling idxmax()/idxmin()
            if z.dropna().empty:
                logger.warning(f"Column '{col}' is empty or all NaN after z-score calculation; skipping anomaly marking.")
                timeseries_df[anomaly_col] = anomaly_flags
                continue
                z = pd.Series(0, index=timeseries_df.index)
            else:
                z = (timeseries_df[col] - mean) / std
            max_idx = z.idxmax()
            min_idx = z.idxmin()

            anomaly_flags[max_idx] = 1
            anomaly_flags[min_idx] = 1

            timeseries_df[anomaly_col] = anomaly_flags

        return timeseries_df


