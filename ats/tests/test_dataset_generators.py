import unittest
import pandas as pd

from ..dataset_generators import HumiTempDatasetGenerator

# Setup logging
from .. import logger
logger.setup()

class TestDatasetGenerator(unittest.TestCase):
    
    def test_generate(self):
        generator = HumiTempDatasetGenerator()
        test_dataset = generator.generate(
            n=12,
            time_span='3D',
            effects=['noise'],
            anomalies=['spike_uv', 'step_uv']
        )
        expected_points = generator._expected_points()
        self.assertEqual(len(test_dataset), 12)
        for i, series in enumerate(test_dataset):
            with self.subTest(dataset=i):
                self.assertIn('temperature', series.columns)
                self.assertIn('humidity', series.columns)
                self.assertEqual(len(series), expected_points)
    
    def test_generate_errors(self):
        generator = HumiTempDatasetGenerator()
        with self.assertRaises(ValueError):
            generator.generate(n=-1,effects=None,anomalies=[])
        with self.assertRaises(ValueError):
            generator.generate(n=0,effects=[],anomalies=[])
        with self.assertRaises(TypeError):
            generator.generate(n='three',effects=[],anomalies=[])
        with self.assertRaises(TypeError):
            generator.generate(effects=[],anomalies='spike_uv')
        with self.assertRaises(TypeError):
            generator.generate(effects=[],anomalies=789)
        with self.assertRaises(ValueError):
            generator.generate(effects=[],n=-3,anomalies=[])
        with self.assertRaises(ValueError):
            generator.generate(effects=[],n=0,anomalies=[])
        with self.assertRaises(TypeError):
            generator.generate(effects='noise',anomalies=[])
        with self.assertRaises(TypeError):
            generator.generate(effects=456,anomalies=[])
        with self.assertRaises(ValueError):
            generator.generate(effects=[],anomalies=['spike_uv', 'spike_mv'])
        with self.assertRaises(ValueError):
            generator.generate(effects=[],anomalies=['clouds'])
        generator.generate(effects=['clouds'],anomalies=['clouds','spike_mv'])  # Should not raise

    def test_generate_random_effects(self):
        generator = HumiTempDatasetGenerator()
        test_dataset = generator.generate(
            n=9,
            time_span='4D',
            random_effects=['clouds'],
            effects=['noise', 'seasons'],
            anomalies=['spike_uv','step_uv']
        )
        self.assertEqual(len(test_dataset), 9)
        for i, series in enumerate(test_dataset, start=1):
            self.assertIsNotNone(series, f"Series {i} is None")
            self.assertTrue(len(series) > 0, f"Series {i} is empty")
    
    def test_no_anomalies(self):
        generator = HumiTempDatasetGenerator()
        test_dataset = generator.generate(
            n=6,
            time_span='2D',
            effects=['noise'],
            anomalies=[]
        )
        self.assertEqual(len(test_dataset), 6)
        for i, series in enumerate(test_dataset):
            with self.subTest(dataset=i):
                self.assertIn('temperature', series.columns)
                self.assertIn('humidity', series.columns)
                self.assertEqual(len(series), generator._expected_points())
                # Verify no anomaly labels are present
                if 'anomaly' in series.columns:
                    self.assertTrue((series['anomaly'] == 0).all() | series['anomaly'].isna().all())
    
    def test_single_anomaly(self):
        generator = HumiTempDatasetGenerator()
        test_dataset = generator.generate(
            n=6,
            time_span='2D',
            effects=['noise'],
            anomalies=['spike_uv']
        )
        self.assertEqual(len(test_dataset), 6)
        for i, series in enumerate(test_dataset):
            with self.subTest(dataset=i):
                self.assertIn('temperature', series.columns)
                self.assertIn('humidity', series.columns)
                self.assertEqual(len(series), generator._expected_points())
                # Verify anomaly labels are either 0 or 1
                if 'anomaly' in series.columns:
                    self.assertTrue(series['anomaly'].isin([0, 1]).all())
                    
    def test_multiple_anomalies(self):
        generator = HumiTempDatasetGenerator()
        test_dataset = generator.generate(
            n=8,
            time_span='2D',
            effects=['noise'],
            anomalies=['spike_uv', 'step_mv']
        )
        self.assertEqual(len(test_dataset), 8)
        for i, series in enumerate(test_dataset):
            with self.subTest(dataset=i):
                self.assertIn('temperature', series.columns)
                self.assertIn('humidity', series.columns)
                self.assertEqual(len(series), generator._expected_points())
                # Verify anomaly labels are either 0, 1, or 2
                if 'anomaly' in series.columns:
                    self.assertTrue(series['anomaly'].isin([0, 1, 2]).all())