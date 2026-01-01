from ..base import TimeseriaAnomalyDetector
from timeseria.models.anomaly_detectors import LSTMAnomalyDetector as TimeseriaLSTMAnomalyDetector  

class LSTMAnomalyDetector(TimeseriaAnomalyDetector):

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

    model_class = TimeseriaLSTMAnomalyDetector
