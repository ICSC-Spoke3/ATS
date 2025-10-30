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