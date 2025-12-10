from ..base import TimeseriaAnomalyDetector
from timeseria.models.anomaly_detectors import LSTMAnomalyDetector as TimeseriaLSTMAnomalyDetector  

class LSTMAnomalyDetector(TimeseriaAnomalyDetector):
    model_class = TimeseriaLSTMAnomalyDetector
