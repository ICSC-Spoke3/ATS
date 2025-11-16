import os
import unittest
import numpy as np
import pandas as pd

from ..anomaly_detectors.naive import MinMaxAnomalyDetector
from ..anomaly_detectors.ml.ifsom import IFSOMAnomalyDetector
from ..anomaly_detectors.stat.robust import _COMNHARAnomalyDetector
from ..utils import generate_timeseries_df
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


class TestStatAnomalyDetectors(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_robust(self):

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




import pandas as pd

def fix_wide_df_isp_format(wide_df, id_col='ID', prefix="IMP_SALDO_CTB_"):
    """
    Convert string date columns of format PREFIX_dd_mm_YYYY
    into proper datetime columns and set ID as index.
    """
    fixed_wide_df = wide_df.copy()

    # Convert date columns
    new_columns = []
    for col in wide_df.columns:
        if col.startswith(prefix):
            date_str = col.replace(prefix, "")
            new_columns.append(pd.to_datetime(date_str, format="%d_%m_%Y"))
        else:
            new_columns.append(col)

    # Assemble
    fixed_wide_df.columns = new_columns

    # Set index but remove the index name
    fixed_wide_df = fixed_wide_df.set_index(id_col)
    fixed_wide_df.index.name = None

    # Ok, return
    return fixed_wide_df.sort_index(axis=1)





class TestMLAnomalyDetectors(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_ifsom(self):

        wide_df_isp_format = pd.read_csv(TEST_DATASETS_PATH + 'ISP_TS_2021-23_minisample_test_small.csv')
        wide_df = fix_wide_df_isp_format(wide_df_isp_format)
        timeseries_df = IFSOMAnomalyDetector.wide_df_to_timeseries_df(wide_df)

        anomaly_detector = IFSOMAnomalyDetector()
        anomaly_detector.fit(timeseries_df)
        results_timeseries_df = anomaly_detector.apply(timeseries_df)

        self.assertEqual(results_timeseries_df.shape, (1095,8))

        self.assertEqual(results_timeseries_df.columns.to_list(), [374107, 1311700, 508010, 602264, '374107_anomaly', '1311700_anomaly', '508010_anomaly', '602264_anomaly'])

        self.assertTrue(results_timeseries_df['374107_anomaly'][0])
        self.assertTrue(results_timeseries_df['374107_anomaly'][-1])

        self.assertFalse(results_timeseries_df['1311700_anomaly'][0])
        self.assertFalse(results_timeseries_df['1311700_anomaly'][-1])

        self.assertFalse(results_timeseries_df['508010_anomaly'][0])
        self.assertFalse(results_timeseries_df['508010_anomaly'][-1])

        self.assertFalse(results_timeseries_df['602264_anomaly'][0])
        self.assertFalse(results_timeseries_df['602264_anomaly'][-1])





