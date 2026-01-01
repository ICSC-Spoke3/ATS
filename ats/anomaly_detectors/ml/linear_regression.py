from ..base import TimeseriaAnomalyDetector
from timeseria.models.anomaly_detectors import LinearRegressionAnomalyDetector as TimeseriaLinearRegressionAnomalyDetector              

class LinearRegressionAnomalyDetector(TimeseriaAnomalyDetector):

    _capabilities = {
        'training': {
            'mode': 'semi-supervised',
            'update': False
        },
        'inference': {
            'streaming': False,
            'dependency': 'window',
            'granularity': 'point-labels'
        },
        'data': {
            'dimensionality': ['univariate-single',
                               'multivariate-single'],
            'sampling': 'regular'
        }
    }

    model_class = TimeseriaLinearRegressionAnomalyDetector