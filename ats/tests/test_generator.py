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

    def test_generate_reference_dataset_randomized_effects(self):
        generator = HumiTempEvaluationDataGenerator()
        reference_dataset = generator.generate_reference_dataset(
            n=5,
            time_span='5D',
            randomize_effects=True,
            effects=['seasons', 'clouds', 'noise']
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
            effects=['noise']
        )
        expected_points = generator._expected_points()

        self.assertEqual(len(test_dataset), 12)
        for i, series in enumerate(test_dataset):
            with self.subTest(dataset=i):
                self.assertIn('temperature', series.columns)
                self.assertIn('humidity', series.columns)
                self.assertEqual(len(series), expected_points)
    
    def test_generate_test_dataset_invalid_n(self):
        generator = HumiTempEvaluationDataGenerator()
        with self.assertRaises(ValueError):
            generator.generate_test_dataset(n=-3)
        with self.assertRaises(ValueError):
            generator.generate_test_dataset(n=0)
        with self.assertRaises(ValueError):
            generator.generate_test_dataset(n=7)  # Not a multiple of 3  

    def test_generate_test_dataset_invalid_effects(self):
        generator = HumiTempEvaluationDataGenerator()
        with self.assertRaises(TypeError):
            generator.generate_test_dataset(effects='seasons')
        with self.assertRaises(TypeError):
            generator.generate_test_dataset(effects=456)
    
    def test_generate_test_dataset_randomized_effects(self):
        generator = HumiTempEvaluationDataGenerator()
        test_dataset = generator.generate_test_dataset(
            n=9,
            time_span='4D',
            randomize_effects=True,
            effects=['noise', 'seasons']
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
            effects=['seasons', 'clouds'],
            randomize_effects=False
        )
        self.assertEqual(len(test_dataset), n)

        expected_points = generator._expected_points()

        groups = [
            test_dataset[0:series_per_group],
            test_dataset[series_per_group:2*series_per_group],
            test_dataset[2*series_per_group:3*series_per_group]
        ]

        for i, group in enumerate(groups):
            self.assertEqual(len(group), series_per_group)