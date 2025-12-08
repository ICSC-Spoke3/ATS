import unittest
import pandas as pd
import numpy as np

from ..anomaly_detectors.basetimeseriamodel import TimeseriaAnomalyDetector, PeriodicAverageAnomalyDetector
from ats.utils import generate_timeseries_df

# Setup logging
from .. import logger
logger.setup()

class TestModels(unittest.TestCase):
    def setUp(self):
        # Generate a sample timeseries DataFrame for testing
        self.data = generate_timeseries_df(entries=200, variables=2, freq='H')

    def test_timeseria_anomaly_detector_not_implemented(self):
        detector = TimeseriaAnomalyDetector()
        with self.assertRaises(NotImplementedError):
            detector.fit(self.data)
        with self.assertRaises(NotImplementedError):
            detector.apply(self.data)

    def test_periodic_average_anomaly_detector(self):
        detector = PeriodicAverageAnomalyDetector()
        detector.fit(self.data)
        applied_series = detector.apply(self.data)

        self.assertIsInstance(applied_series, pd.DataFrame)
        self.assertIn('anomaly', applied_series.columns)