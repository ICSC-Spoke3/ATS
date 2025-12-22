import unittest
import pandas as pd
import numpy as np

from ..anomaly_detectors.base import TimeseriaAnomalyDetector
from ..anomaly_detectors.stat.periodic_average import PeriodicAverageAnomalyDetector
from ..anomaly_detectors.stat.arima import ARIMAAnomalyDetector
from ..anomaly_detectors.ml.prophet import ProphetAnomalyDetector
from ats.utils import generate_timeseries_df

# Setup logging
from .. import logger
logger.setup()

class TestTimeseriaAnomalyDetectors(unittest.TestCase):
    def setUp(self):
        # Generate a sample timeseries DataFrame for testing
        self.data_mv = generate_timeseries_df(entries=200, variables=2, freq='H')
        self.data_uv = generate_timeseries_df(entries=200, variables=1, freq='H')

    def test_timeseria_anomaly_detector_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            detector = TimeseriaAnomalyDetector()
            
    def test_periodic_average_anomaly_detector(self):
        detector = PeriodicAverageAnomalyDetector()
        detector.fit(self.data_mv)
        applied_series = detector.apply(self.data_mv)

        self.assertIsInstance(applied_series, pd.DataFrame)
        self.assertIn('anomaly', applied_series.columns)
    
    def test_apply_params(self):
        detector = PeriodicAverageAnomalyDetector()
        detector.set_apply_params(threshold=2.0)
        params = detector.get_apply_params()
        
        self.assertIn('threshold', params)
        self.assertEqual(params['threshold'], 2.0)
    
    def test_prophet_anomaly_detector(self):
        detector = ProphetAnomalyDetector()
        detector.fit(self.data_uv)
        applied_series = detector.apply(self.data_uv)
        self.assertIsInstance(applied_series, pd.DataFrame)
        self.assertIn('anomaly', applied_series.columns)
    
    def test_arima_anomaly_detector(self):
        detector = ARIMAAnomalyDetector()
        detector.fit(self.data_uv)
        applied_series = detector.apply(self.data_uv)
        self.assertIsInstance(applied_series, pd.DataFrame)
        self.assertIn('anomaly', applied_series.columns)