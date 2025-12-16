import os
import unittest
import numpy as np
import pandas as pd

from ..anomaly_detectors.naive import MinMaxAnomalyDetector, ZScoreAnomalyDetector
from ..anomaly_detectors.ml.ifsom import IFSOMAnomalyDetector
from ..anomaly_detectors.stat.robust import _COMNHARAnomalyDetector
from ..utils import generate_timeseries_df, load_isp_format_wide_df, wide_df_to_list_of_timeseries_df, timeseries_df_to_list_of_timeseries_df
from ..anomaly_detectors.stat.support_functions import generate_contaminated_dataframe

# Setup logging
from .. import logger
logger.setup()

TEST_DATASETS_PATH = os.path.join(os.path.dirname(__file__), 'test_data', '')

class TestNaiveAnomalyDetectors(unittest.TestCase):

    def test_minmax(self):

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

    def test_zscore(self):

        anomaly_detector = ZScoreAnomalyDetector()
        timeseries_df = generate_timeseries_df(entries=10, variables=2)
        timeseries_df_scored = anomaly_detector.apply(timeseries_df)

        self.assertEqual(timeseries_df_scored.shape, (10,4))

        self.assertEqual(timeseries_df_scored.loc['2025-06-10 14:00:00+00:00', 'value_1_anomaly'], 0)
        self.assertEqual(timeseries_df_scored.loc['2025-06-10 23:00:00+00:00', 'value_2_anomaly'], 0)

        self.assertEqual(timeseries_df_scored.loc['2025-06-10 19:00:00+00:00', 'value_1_anomaly'], 1)
        self.assertEqual(timeseries_df_scored.loc['2025-06-10 19:00:00+00:00', 'value_1_anomaly'], 1)

        self.assertEqual(timeseries_df_scored.loc['2025-06-10 18:00:00+00:00', 'value_2_anomaly'], 1)
        self.assertEqual(timeseries_df_scored.loc['2025-06-10 21:00:00+00:00', 'value_2_anomaly'], 1)


class TestStatAnomalyDetectors(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_robust_on_multivariate(self):

        # Generate data with anomalies
        timeseries_df, _ = generate_contaminated_dataframe(cn=150, cd=10, prct=0.4, cnj=80, tim=-4, repr=True)

        # Instantiate the anomaly detector
        anomaly_detector = _COMNHARAnomalyDetector()
        results_timeseries_df = anomaly_detector.apply(timeseries_df)

        # Uncomment to inspect results
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #    print(results_timeseries_df)

        self.assertFalse(results_timeseries_df.iloc[0]['anomaly'])
        self.assertFalse(results_timeseries_df.iloc[79]['anomaly'])
        self.assertTrue(results_timeseries_df.iloc[80]['anomaly'])
        self.assertTrue(results_timeseries_df.iloc[81]['anomaly'])
        self.assertFalse(results_timeseries_df.iloc[82]['anomaly'])


    def test_robust_on_list_of_univariate(self):

        # Generate data with anomalies
        timeseries_df, _ = generate_contaminated_dataframe(cn=150, cd=10, prct=0.4, cnj=80, tim=-4, repr=True)

        # Convert a big multivariate to n signle univariate
        list_of_timeseries_df = timeseries_df_to_list_of_timeseries_df(timeseries_df)

        # Instantiate the anomaly detector
        anomaly_detector = _COMNHARAnomalyDetector()
        results_list_timeseries_df = anomaly_detector.apply(list_of_timeseries_df)

        # Uncomment to inspect results
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #    print(results_list_timeseries_df)

        self.assertEqual(len(results_list_timeseries_df),10)
        self.assertFalse(results_list_timeseries_df[0].iloc[0]['anomaly'])
        self.assertFalse(results_list_timeseries_df[0].iloc[79]['anomaly'])
        self.assertTrue(results_list_timeseries_df[0].iloc[80]['anomaly'])
        self.assertTrue(results_list_timeseries_df[0].iloc[81]['anomaly'])
        self.assertFalse(results_list_timeseries_df[0].iloc[82]['anomaly'])
        self.assertFalse(results_list_timeseries_df[0].iloc[-1]['anomaly'])


class TestMLAnomalyDetectors(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_ifsom(self):

        wide_df = load_isp_format_wide_df(TEST_DATASETS_PATH + 'ISP_TS_2021-23_minisample_test_small.csv')

        list_of_timeseries_df = wide_df_to_list_of_timeseries_df(wide_df)

        anomaly_detector = IFSOMAnomalyDetector()
        anomaly_detector.fit(list_of_timeseries_df)

        list_of_results_timeseries_df = anomaly_detector.apply(list_of_timeseries_df)

        self.assertEqual(len(list_of_results_timeseries_df), 4)
        self.assertEqual(list_of_results_timeseries_df[0].shape, (1095,2))

        self.assertEqual(list_of_results_timeseries_df[0].columns.to_list(), ['374107', 'anomaly'])
        self.assertEqual(list_of_results_timeseries_df[1].columns.to_list(), ['1311700', 'anomaly'])
        self.assertEqual(list_of_results_timeseries_df[2].columns.to_list(), ['508010', 'anomaly'])
        self.assertEqual(list_of_results_timeseries_df[3].columns.to_list(), ['602264', 'anomaly'])

        self.assertTrue(list_of_results_timeseries_df[0]['anomaly'][0])
        self.assertTrue(list_of_results_timeseries_df[0]['anomaly'][-1])

        self.assertFalse(list_of_results_timeseries_df[1]['anomaly'][0])
        self.assertFalse(list_of_results_timeseries_df[1]['anomaly'][-1])

        self.assertFalse(list_of_results_timeseries_df[2]['anomaly'][0])
        self.assertFalse(list_of_results_timeseries_df[2]['anomaly'][-1])

        self.assertFalse(list_of_results_timeseries_df[3]['anomaly'][0])
        self.assertFalse(list_of_results_timeseries_df[3]['anomaly'][-1])

