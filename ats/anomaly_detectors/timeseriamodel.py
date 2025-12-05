from ats.utils import convert_timeseries_df_to_timeseries, convert_timeseries_to_timeseries_df
from .base import AnomalyDetector
from timeseria.datastructures import TimeSeries
from timeseria.models.anomaly_detectors import PeriodicAverageAnomalyDetector
import pandas as pd

# Setup logging
import logging
logger = logging.getLogger(__name__)

class TimeseriaAnomalyDetector(AnomalyDetector):  
    
    def __init__(self):
        self.model = PeriodicAverageAnomalyDetector()
        pass

    def apply(self,data):
        if not isinstance(data,pd.DataFrame):
            raise NotImplementedError()
        timeseries_df = data

        # Using timeseria to fit and apply the model
        timeseries = convert_timeseries_df_to_timeseries(timeseries_df)
        model = self.model
        model.fit(timeseries)
        timeseries = model.apply(timeseries)

        # Convert back to DataFrame
        timeseries_df = convert_timeseries_to_timeseries_df(timeseries)
        for col in timeseries_df.columns:
            anomaly_col = f"{col}_anomaly"
            timeseries_df[anomaly_col] = timeseries_df[f"{col}_anomaly"]
            
        return timeseries_df
    
    # Implementare fit e apply
    # Aggiungere value_1_anomaly value_2_anomaly anomaly 
    # anomaly = unione (?)
    #timeseries --> df