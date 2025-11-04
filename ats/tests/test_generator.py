import unittest
import pandas as pd

from ..generators import HumiTempEvaluationDataGenerator

# Setup logging
from .. import logger
logger.setup()

class TestGenerator(unittest.TestCase):

    def test_generate_reference_dataset(self):
        generator = HumiTempEvaluationDataGenerator()
        reference_dataset = generator.generate_reference_dataset(time_span='5D', effects=None)
        expected_points = generator._expected_points()

        self.assertEqual(len(reference_dataset), 3)
        for i, series in enumerate(reference_dataset):
            with self.subTest(dataset=i):
                self.assertIn('temperature', series.columns)
                self.assertIn('humidity', series.columns)
                self.assertEqual(len(series), expected_points)
    
    def test_generate_reference_dataset_invalid_n(self):
        generator = HumiTempEvaluationDataGenerator()
        with self.assertRaises(ValueError):
            generator.generate_reference_dataset(n=-1,effects=None)
        with self.assertRaises(ValueError):
            generator.generate_reference_dataset(n=0,effects=[])
        with self.assertRaises(ValueError):
            generator.generate_reference_dataset(n='three',effects=[])
    
    def test_generate_reference_dataset_invalid_effects(self):
        generator = HumiTempEvaluationDataGenerator()
        with self.assertRaises(TypeError):
            generator.generate_reference_dataset(effects='noise')
        with self.assertRaises(TypeError):
            generator.generate_reference_dataset(effects=123)

    def test_generate_reference_dataset_random_effects(self):
        generator = HumiTempEvaluationDataGenerator()
        reference_dataset = generator.generate_reference_dataset(
            n=5,
            time_span='5D',
            random_effects=['clouds'],
            effects=['seasons', 'noise']
        )
        self.assertEqual(len(reference_dataset), 5)
        for i, series in enumerate(reference_dataset, start=1):
            self.assertIsNotNone(series, f"Series {i} is None")
            self.assertTrue(len(series) > 0, f"Series {i} is empty")

    # Beginning of test for generate_test_dataset
    def test_generate_test_dataset(self):
        generator = HumiTempEvaluationDataGenerator()
        test_dataset = generator.generate_test_dataset(
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
    
    def test_generate_test_dataset_invalid_anomalies(self):
        generator = HumiTempEvaluationDataGenerator()
        with self.assertRaises(NotImplementedError):
            generator.generate_test_dataset(anomalies='spike_uv')
        with self.assertRaises(TypeError):
            generator.generate_test_dataset(anomalies=789)
        with self.assertRaises(NotImplementedError):
            generator.generate_test_dataset(random_anomalies=['random_spike']) 
        with self.assertRaises(ValueError):
            generator.generate_test_dataset(anomalies=['spike_uv'])
        with self.assertRaises(ValueError):
            generator.generate_test_dataset(n=-3)
        with self.assertRaises(ValueError):
            generator.generate_test_dataset(n=0)
        with self.assertRaises(ValueError):
            generator.generate_test_dataset(n=7)  # Not a multiple of 3  
        with self.assertRaises(TypeError):
            generator.generate_test_dataset(effects='seasons')
        with self.assertRaises(TypeError):
            generator.generate_test_dataset(effects=456)

    def test_generate_test_dataset_random_effects(self):
        generator = HumiTempEvaluationDataGenerator()
        test_dataset = generator.generate_test_dataset(
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

def test_generate_test_dataset_group(self):
    generator = HumiTempEvaluationDataGenerator()
    n = 6
    num_groups = 3
    series_per_group = n // num_groups
    test_dataset = generator.generate_test_dataset(
        n=n,
        time_span='2D',
        effects=['noise'],
        anomalies=['spike_uv', 'step_uv']
    )
    self.assertEqual(len(test_dataset), n)
    group_rules = {
            0: lambda cols: 'spike_uv' not in cols and 'step_uv' not in cols,                  # Nessuna anomalia
            1: lambda cols: ('spike_uv' in cols) ^ ('step_uv' in cols),                       # Solo una delle due
            2: lambda cols: 'spike_uv' in cols and 'step_uv' in cols,                         # Entrambe presenti
        }
    for group in range(num_groups):
        start_idx = group * series_per_group
        end_idx = start_idx + series_per_group
        for i in range(start_idx, end_idx):
            with self.subTest(group=group, dataset=i):
                series = test_dataset[i]
                cols = series.columns
                self.assertTrue(
                    group_rules[group](cols),(
                        f"Group {group}, dataset {i}: unexpected anomaly combination. "
                        f"Columns: {list(cols)}"))

