from ats.utils import convert_timeseries_df_to_timeseries, convert_timeseries_to_timeseries_df
from .base import AnomalyDetector
from timeseria.datastructures import TimeSeries
from timeseria.models.anomaly_detectors import PeriodicAverageAnomalyDetector as TimeseriaPeriodicAverageAnomalyDetector
from timeseria.models.anomaly_detectors import LSTMAnomalyDetector as TimeseriaLSTMAnomalyDetector  
from timeseria.models.anomaly_detectors import LinearRegressionAnomalyDetector as TimeseriaLinearRegressionAnomalyDetector              
import pandas as pd

# Setup logging
import logging
logger = logging.getLogger(__name__)

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

class PeriodicAverageAnomalyDetector(TimeseriaAnomalyDetector):
    model_class = TimeseriaPeriodicAverageAnomalyDetector

class LSTMAnomalyDetector(TimeseriaAnomalyDetector):
    model_class = TimeseriaLSTMAnomalyDetector

class LinearRegressionAnomalyDetector(TimeseriaAnomalyDetector):
    model_class = TimeseriaLinearRegressionAnomalyDetector