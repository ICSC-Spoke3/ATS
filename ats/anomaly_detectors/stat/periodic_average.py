from ..base import TimeseriaAnomalyDetector
from timeseria.models.anomaly_detectors import PeriodicAverageAnomalyDetector as TimeseriaPeriodicAverageAnomalyDetector

class PeriodicAverageAnomalyDetector(TimeseriaAnomalyDetector):

    capabilities = {
        'mode': 'semi-supervised',
        'streaming': True,
        'context': 'window',
        'granularity': 'point',
        'multivariate': True,
        'scope': 'agnostic'
    }

    model_class = TimeseriaPeriodicAverageAnomalyDetector
