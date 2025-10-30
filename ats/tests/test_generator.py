import unittest
import pandas as pd

from ..generators import HumiTempEvaluationDataGenerator

# Setup logging
from .. import logger
logger.setup()

class TestGenerator(unittest.TestCase):

    def test_generate_reference_dataset(self):
        generator = HumiTempEvaluationDataGenerator()
        reference_dataset = generator.generate_reference_dataset(observation_window='5D')
        expected_points = generator.__expected_points__()

        self.assertEqual(len(reference_dataset), 3)
        for i, series in enumerate(reference_dataset):
            with self.subTest(dataset=i):
                self.assertIn('temperature', series.columns)
                self.assertIn('humidity', series.columns)
                self.assertEqual(len(series), expected_points)
    
    def test_generate_reference_dataset_invalid_howmany_series(self):
        generator = HumiTempEvaluationDataGenerator()
        with self.assertRaises(ValueError):
            generator.generate_reference_dataset(howmany_series=-1)
        with self.assertRaises(ValueError):
            generator.generate_reference_dataset(howmany_series=0)
        with self.assertRaises(ValueError):
            generator.generate_reference_dataset(howmany_series='three')
    
    def test_generate_reference_dataset_invalid_effects(self):
        generator = HumiTempEvaluationDataGenerator()
        with self.assertRaises(TypeError):
            generator.generate_reference_dataset(effects='noise')
        with self.assertRaises(TypeError):
            generator.generate_reference_dataset(effects=123)

    def test_generate_reference_dataset_randomized_effects(self):
        generator = HumiTempEvaluationDataGenerator()
        reference_dataset = generator.generate_reference_dataset(
            howmany_series=5,
            observation_window='5D',
            randomize_effects=True
        )
        self.assertEqual(len(reference_dataset), 5)
        for i, series in enumerate(reference_dataset, start=1):
            self.assertIsNotNone(series, f"Series {i} is None")
            self.assertTrue(len(series) > 0, f"Series {i} is empty")
        
    def test_generate_test_dataset(self):
        generator = HumiTempEvaluationDataGenerator()
        test_dataset = generator.generate_test_dataset(
            howmany_series=12,
            observation_window='3D',
            effects=['noise']
        )
        expected_points = generator.__expected_points__()

        self.assertEqual(len(test_dataset), 12)
        for i, series in enumerate(test_dataset):
            with self.subTest(dataset=i):
                self.assertIn('temperature', series.columns)
                self.assertIn('humidity', series.columns)
                self.assertEqual(len(series), expected_points)
    
    def test_generate_test_dataset_invalid_howmany_series(self):
        generator = HumiTempEvaluationDataGenerator()
        with self.assertRaises(ValueError):
            generator.generate_test_dataset(howmany_series=-3)
        with self.assertRaises(ValueError):
            generator.generate_test_dataset(howmany_series=0)
        with self.assertRaises(ValueError):
            generator.generate_test_dataset(howmany_series=7)  # Not a multiple of 3    ù

    def test_generate_test_dataset_invalid_effects(self):
        generator = HumiTempEvaluationDataGenerator()
        with self.assertRaises(TypeError):
            generator.generate_test_dataset(effects='seasons')
        with self.assertRaises(TypeError):
            generator.generate_test_dataset(effects=456)
    
    def test_generate_test_dataset_randomized_effects(self):
        generator = HumiTempEvaluationDataGenerator()
        test_dataset = generator.generate_test_dataset(
            howmany_series=9,
            observation_window='4D',
            randomize_effects=True
        )
        self.assertEqual(len(test_dataset), 9)
        for i, series in enumerate(test_dataset, start=1):
            self.assertIsNotNone(series, f"Series {i} is None")
            self.assertTrue(len(series) > 0, f"Series {i} is empty")

    def test_generate_test_dataset_cluster(self):
        generator = HumiTempEvaluationDataGenerator()
        
        howmany_series = 6
        num_clusters = 3
        series_per_cluster = howmany_series // num_clusters
        
        test_dataset = generator.generate_test_dataset(
            howmany_series=howmany_series,
            observation_window='2D',
            effects=['seasons', 'clouds'],
            randomize_effects=False
        )
        self.assertEqual(len(test_dataset), howmany_series)

        expected_points = generator.__expected_points__()

        clusters = [
            test_dataset[0:series_per_cluster],
            test_dataset[series_per_cluster:2*series_per_cluster],
            test_dataset[2*series_per_cluster:3*series_per_cluster]
        ]

        for i, cluster in enumerate(clusters):
            self.assertEqual(len(cluster), series_per_cluster)