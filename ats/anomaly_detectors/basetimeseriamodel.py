from ats.utils import convert_timeseries_df_to_timeseries, convert_timeseries_to_timeseries_df
from .base import AnomalyDetector
from timeseria.datastructures import TimeSeries
from timeseria.models.anomaly_detectors import PeriodicAverageAnomalyDetector as TimeseriaPeriodicAverageAnomalyDetector
import pandas as pd

# Setup logging
import logging
logger = logging.getLogger(__name__)

class TimeseriaAnomalyDetector(AnomalyDetector):  
    
    def __init__(self):
        self.model = None
    
    def fit(self, data, *args, **kwargs):
        if not isinstance(data,pd.DataFrame):
            raise NotImplementedError('Not yet implemented for non DataFrame inputs')
        timeseries_df = data

        # Using timeseria to fit the model
        timeseries = convert_timeseries_df_to_timeseries(timeseries_df)
        model = self.model
        model.fit(timeseries, *args, **kwargs)
        self.model = model

    def apply(self, data, *args, **kwargs):
        if not isinstance(data,pd.DataFrame):
            raise NotImplementedError('Not yet implemented for non DataFrame inputs')
        timeseries_df = data

        # Using timeseria to fit and apply the model
        timeseries = convert_timeseries_df_to_timeseries(timeseries_df)
        model = self.model

        #TODO: remove fit from apply at some point
        model.fit(timeseries, *args, **kwargs)
        timeseries = model.apply(timeseries, *args, **kwargs)

        # Convert back to DataFrame
        timeseries_df = convert_timeseries_to_timeseries_df(timeseries)
            
        return timeseries_df

class PeriodicAverageAnomalyDetector(TimeseriaAnomalyDetector):
    def __init__(self):
        super().__init__()
        self.model = TimeseriaPeriodicAverageAnomalyDetector()

    def apply(self,data):
        return super().apply(data)
