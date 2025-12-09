from .base import AnomalyDetector, TimeseriaAnomalyDetector
from timeseria.models.anomaly_detectors import PeriodicAverageAnomalyDetector as TimeseriaPeriodicAverageAnomalyDetector
from timeseria.models.anomaly_detectors import LSTMAnomalyDetector as TimeseriaLSTMAnomalyDetector  
from timeseria.models.anomaly_detectors import LinearRegressionAnomalyDetector as TimeseriaLinearRegressionAnomalyDetector              

class PeriodicAverageAnomalyDetector(TimeseriaAnomalyDetector):
    model_class = TimeseriaPeriodicAverageAnomalyDetector

class LSTMAnomalyDetector(TimeseriaAnomalyDetector):
    model_class = TimeseriaLSTMAnomalyDetector

class LinearRegressionAnomalyDetector(TimeseriaAnomalyDetector):
    model_class = TimeseriaLinearRegressionAnomalyDetector