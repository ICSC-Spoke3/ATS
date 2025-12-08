import unittest

from ..anomaly_detectors.basetimeseriamodel import TimeseriaAnomalyDetector, PeriodicAverageAnomalyDetector

# Setup logging
from .. import logger
logger.setup()

class TestModels(unittest.TestCase):