import unittest
import pandas as pd

from ..anomaly_detectors import MinMaxAnomalyDetector
from ..utils import generate_timeseries_df

# Setup logging
from .. import logger
logger.setup()

class TestMinMaxAnomalyDetector(unittest.TestCase):

    def test_univariate(self):

        anomaly_detector = MinMaxAnomalyDetector()
        timeseries_df = generate_timeseries_df(entries=10, variables=2)
        timeseries_df_scored = anomaly_detector.apply(timeseries_df)

        self.assertEqual(timeseries_df_scored.shape, (10,4))

        #                             value_1   value_2  value_1_anomaly  value_2_anomaly
        # timestamp
        # 2025-06-10 14:00:00+00:00  0.000000  0.707107                0                0
        # 2025-06-10 15:00:00+00:00  0.841471  0.977061                0                0
        # 2025-06-10 16:00:00+00:00  0.909297  0.348710                0                0
        # 2025-06-10 17:00:00+00:00  0.141120 -0.600243                0                0
        # 2025-06-10 18:00:00+00:00 -0.756802 -0.997336                0                1
        # 2025-06-10 19:00:00+00:00 -0.958924 -0.477482                1                0
        # 2025-06-10 20:00:00+00:00 -0.279415  0.481366                0                0
        # 2025-06-10 21:00:00+00:00  0.656987  0.997649                0                1
        # 2025-06-10 22:00:00+00:00  0.989358  0.596698                1                0
        # 2025-06-10 23:00:00+00:00  0.412118 -0.352855                0                0

        self.assertEqual(timeseries_df_scored.loc['2025-06-10 14:00:00+00:00', 'value_1_anomaly'], 0)
        self.assertEqual(timeseries_df_scored.loc['2025-06-10 23:00:00+00:00', 'value_2_anomaly'], 0)

        self.assertEqual(timeseries_df_scored.loc['2025-06-10 19:00:00+00:00', 'value_1_anomaly'], 1)
        self.assertEqual(timeseries_df_scored.loc['2025-06-10 19:00:00+00:00', 'value_1_anomaly'], 1)

        self.assertEqual(timeseries_df_scored.loc['2025-06-10 18:00:00+00:00', 'value_2_anomaly'], 1)
        self.assertEqual(timeseries_df_scored.loc['2025-06-10 21:00:00+00:00', 'value_2_anomaly'], 1)
