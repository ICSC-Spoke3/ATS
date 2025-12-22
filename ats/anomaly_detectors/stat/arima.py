from timeseria.models.forecasters import ARIMAForecaster
from timeseria.models.anomaly_detectors import ModelBasedAnomalyDetector
from ..base import TimeseriaAnomalyDetector

class _ArimaBaseDetector(ModelBasedAnomalyDetector):
    model_class = ARIMAForecaster

class ARIMAAnomalyDetector(TimeseriaAnomalyDetector):
    model_class = _ArimaBaseDetector

    #TODO: define right capabilities
    capabilities = {
        'mode': 'semi-supervised',
        'streaming': True,
        'context': 'window',
        'granularity': 'point',
        'multivariate': False,
        'scope': 'agnostic'
    }
    
