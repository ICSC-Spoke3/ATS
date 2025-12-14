from ..base import TimeseriaAnomalyDetector
from timeseria.models.anomaly_detectors import LinearRegressionAnomalyDetector as TimeseriaLinearRegressionAnomalyDetector              

class LinearRegressionAnomalyDetector(TimeseriaAnomalyDetector):

    capabilities = {
        'mode': 'semi-supervised',
        'streaming': True,
        'context': 'window',
        'granularity': 'point',
        'multivariate': False,
        'scope': 'agnostic'
    }

    model_class = TimeseriaLinearRegressionAnomalyDetector