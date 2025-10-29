import unittest
import pandas as pd

from ..generator import HumiTempEvaluationDataGenerator

# Setup logging
from .. import logger
logger.setup()

class TestGenerator(unittest.TestCase):

    def test_generate_reference_datasets(self):
        generator = HumiTempEvaluationDataGenerator()
        reference_datasets = generator.generate_reference_datasets()
        expected_points = generator.__expected_points__()

        self.assertEqual(len(reference_datasets), 3)
        for i, dataset in enumerate(reference_datasets):
            with self.subTest(dataset=i):
                self.assertIn('temperature', dataset.columns)
                self.assertIn('humidity', dataset.columns)
                self.assertEqual(len(dataset), expected_points)