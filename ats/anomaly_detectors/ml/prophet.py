from timeseria.models.forecasters import ProphetForecaster
from timeseria.models.anomaly_detectors import ModelBasedAnomalyDetector
from ..base import TimeseriaAnomalyDetector

class _ProphetBaseDetector(ModelBasedAnomalyDetector):
    model_class = ProphetForecaster

class ProphetAnomalyDetector(TimeseriaAnomalyDetector):
    model_class = _ProphetBaseDetector
    
    #TODO: define right capabilities
    capabilities = {
        'mode': 'semi-supervised',
        'streaming': True,
        'context': 'window',
        'granularity': 'point',
        'multivariate': False,
        'scope': 'agnostic'
    }
    