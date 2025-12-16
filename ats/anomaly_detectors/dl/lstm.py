from ..base import TimeseriaAnomalyDetector
from timeseria.models.anomaly_detectors import LSTMAnomalyDetector as TimeseriaLSTMAnomalyDetector  

class LSTMAnomalyDetector(TimeseriaAnomalyDetector):

    capabilities = {
        'mode': 'semi-supervised',
        'streaming': True,
        'context': 'window',
        'granularity': 'point',
        'multivariate': True,
        'scope': 'agnostic'
    }

    model_class = TimeseriaLSTMAnomalyDetector
