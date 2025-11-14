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
            generator.generate(effects=[],random_effects=['random_spike'],anomalies=[]) 
        with self.assertRaises(ValueError):
            generator.generate(effects=[],anomalies=['spike_uv'])
        with self.assertRaises(ValueError):
            generator.generate(effects=[],n=-3,anomalies=[])
        with self.assertRaises(ValueError):
            generator.generate(effects=[],n=0,anomalies=[])
        with self.assertRaises(ValueError):
            generator.generate(effects=[],n=7,anomalies=[] )  # Not a multiple of 3  
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

    def test_generate_group(self):
        generator = HumiTempDatasetGenerator()
        n = 6
        num_groups = 3
        series_per_group = n // num_groups
        test_dataset = generator.generate(
            n=n,
            time_span='2D',
            effects=['noise'],
            anomalies=['spike_uv', 'step_uv']
        )
        self.assertEqual(len(test_dataset), n)
        group_rules = {
                0: lambda series: ('anomaly_label' not in series.columns) or (series['anomaly_label'].isna().all()),
                1: lambda series: 'anomaly_label' in series.columns and len(series['anomaly_label'].dropna().unique()) == 1,
                2: lambda series: 'anomaly_label' in series.columns and len(series['anomaly_label'].dropna().unique()) == 2,
            }
        for group in range(num_groups):
            start_idx = group * series_per_group
            end_idx = start_idx + series_per_group
            for i in range(start_idx, end_idx):
                with self.subTest(group=group, dataset=i):
                    series = test_dataset[i]
                    self.assertTrue(
                        group_rules[group](series),(
                            f"Group {group}, dataset {i}: unexpected anomaly combination. "
                            f"Columns: {list(series.columns)}"))

