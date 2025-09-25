from ..evaluators import evaluate_anomaly_detector
from ..anomaly_detectors import MinMaxAnomalyDetector
from ..utils import generate_timeseries_df
import unittest
import pandas as pd


# Setup logging
from .. import logger
logger.setup()

class TestEvaluators(unittest.TestCase):

    def test_evaluate_anomaly_detector(self):

        min_max_anomaly_detector = MinMaxAnomalyDetector()
        timeseries_df = generate_timeseries_df(entries=4, variables=1)
        timeseries_df['anomaly_label'] = ['anomaly_1','normal','anomaly_2','normal']
        # Generated DataFrame:
        #                             value                  anomaly_label
        # timestamp
        # 2025-06-10 14:00:00+00:00   0.0                    anomaly_1
        # 2025-06-10 15:00:00+00:00   0.8414709848078965     normal
        # 2025-06-10 16:00:00+00:00   0.9092974268256817     anomaly_2
        # 2025-06-10 17:00:00+00:00   0.1411200080598672     normal
        
        evaluation_results = evaluate_anomaly_detector(min_max_anomaly_detector, timeseries_df)
        # Evaluation_results
        #{ 'normal':     value_anomaly    False
        #                Name: 2025-06-10 15:00:00+00:00, dtype: bool,
        #  'anomaly_1':  value_anomaly    True
        #                Name: 2025-06-10 14:00:00+00:00, dtype: bool,
        # 'anomaly_2':   value_anomaly    True
        #                Name: 2025-06-10 16:00:00+00:00, dtype: bool}

        self.assertIsInstance(evaluation_results,dict)
        self.assertEqual(len(evaluation_results),3)
        self.assertIsInstance(evaluation_results['anomaly_1'],pd.Series)
        self.assertEqual(evaluation_results['anomaly_2'].loc['value_anomaly'],True)
        self.assertEqual(evaluation_results['anomaly_1'].loc['value_anomaly'],True)
        self.assertEqual(evaluation_results['normal'].loc['value_anomaly'],False)






  