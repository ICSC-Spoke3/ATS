import unittest

from ..generator import HumiTempEvaluationDataGenerator

# Setup logging
from .. import logger
logger.setup()

class TestGenerator(unittest.TestCase):

    def test_generate_reference_data(self):
        data_size = 100
        effect_type = ['temperature', 'humidity']
        reference_data = HumiTempEvaluationDataGenerator.generate_reference_data(data_size, effect_type)
        self.assertEqual(len(reference_data), data_size)
        self.assertIn('temperature', reference_data.columns)
        self.assertIn('humidity', reference_data.columns)