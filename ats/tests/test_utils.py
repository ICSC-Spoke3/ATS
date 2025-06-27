import unittest
import pandas as pd

from ..utils import generate_timeseries_df

# Setup logging
from .. import logger
logger.setup()

class TestUtils(unittest.TestCase):

    def test_generate_timeseries_df(self):

        timeseries_df = generate_timeseries_df(entries=10, variables=2)
        self.assertEqual(timeseries_df.shape, (10,2))

