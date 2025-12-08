import unittest
import pandas as pd
import numpy as np

from ..anomaly_detectors.basetimeseriamodel import TimeseriaAnomalyDetector, PeriodicAverageAnomalyDetector

# Setup logging
from .. import logger
logger.setup()

class TestModels(unittest.TestCase):
    def test_timeseria_anomaly_detector_not_implemented(self):
        detector = TimeseriaAnomalyDetector()
        with self.assertRaises(NotImplementedError):
            detector.fit(data=None)
        with self.assertRaises(NotImplementedError):
            detector.apply(data=None)

    def test_periodic_average_anomaly_detector(self):
        detector = PeriodicAverageAnomalyDetector()
        # Create a simple DataFrame for testing
        date_range = pd.date_range(start='2023-01-01', periods=100, freq='H')
        data = pd.DataFrame({
            'timestamp': date_range,
            'value': np.random.rand(100)
        }).set_index('timestamp')

        detector.fit(data)
        serie_fitted = detector.apply(data)

        self.assertIsInstance(serie_fitted, pd.DataFrame)
        self.assertEqual(serie_fitted.shape, data.shape)