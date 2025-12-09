from ..base import TimeseriaAnomalyDetector
from timeseria.models.anomaly_detectors import LinearRegressionAnomalyDetector as TimeseriaLinearRegressionAnomalyDetector              

class LinearRegressionAnomalyDetector(TimeseriaAnomalyDetector):
    model_class = TimeseriaLinearRegressionAnomalyDetector