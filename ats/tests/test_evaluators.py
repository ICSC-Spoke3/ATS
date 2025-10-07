from ..evaluators import evaluate_anomaly_detector
from ..anomaly_detectors.naive import MinMaxAnomalyDetector
from ..synthetic_data import SyntheticHumiTempTimeseriesGenerator
from ..utils import generate_timeseries_df
import unittest
import pandas as pd
import random as rnd
import numpy as np


# Setup logging
from .. import logger
logger.setup()

class TestEvaluators(unittest.TestCase):

    def setUp(self):

        rnd.seed(123)
        np.random.seed(123)

    def test_evaluate_anomaly_detector(self):

        min_max_anomaly_detector = MinMaxAnomalyDetector()
        timeseries_df = generate_timeseries_df(entries=4, variables=1)
        timeseries_df['anomaly_label'] = ['anomaly_1', None,'anomaly_2', None]
        # Generated DataFrame:
        #                             value                  anomaly_label
        # timestamp
        # 2025-06-10 14:00:00+00:00   0.0                    'anomaly_1'
        # 2025-06-10 15:00:00+00:00   0.8414709848078965      None
        # 2025-06-10 16:00:00+00:00   0.9092974268256817     'anomaly_2'
        # 2025-06-10 17:00:00+00:00   0.1411200080598672      None
        
        evaluation_results = evaluate_anomaly_detector(min_max_anomaly_detector, timeseries_df)
        # Evaluation_results:
        #{ 'false_positives': 0,
        #  'anomaly_1':       True,
        #  'anomaly_2':       True
        #}

        self.assertIsInstance(evaluation_results,dict)
        self.assertEqual(len(evaluation_results),3)
        self.assertIn('anomaly_1',evaluation_results.keys())
        self.assertIn('anomaly_2',evaluation_results.keys())
        self.assertIn('false_positives',evaluation_results.keys())
        self.assertIsInstance(evaluation_results['anomaly_1'],bool)
        self.assertIsInstance(evaluation_results['anomaly_2'],bool)
        self.assertIsInstance(evaluation_results['false_positives'],int)
        self.assertEqual(evaluation_results['anomaly_2'],True)
        self.assertEqual(evaluation_results['anomaly_1'],True)
        self.assertEqual(evaluation_results['false_positives'],0)

    def test_evaluate_anomaly_det_on_spiked_synth_timeseries(self):

        spiked_humi_temp_generator = SyntheticHumiTempTimeseriesGenerator()
        humi_temp_df = spiked_humi_temp_generator.generate(anomalies=['spike_uv'],effects=[])
        # Generated DataFrame:
        # Timestamp                    temperature           humidity             anomaly_label    effect_label
        # ...
        # 1973-05-07 11:02:00+00:00    24.761107302835810    50.63704719243785     None            None
        # 1973-05-07 11:17:00+00:00    24.868377982941322    50.350992045489804    None            None
        # 1973-05-07 11:32:00+00:00    24.944096137309916    50.14907696717356     None            None
        # 1973-05-07 11:47:00+00:00    15.987937529180115    59.03216658885302     spike_uv        None
        # 1973-05-07 12:02:00+00:00    24.999714422981285    50.000761538716574    None            None
        # 1973-05-07 12:17:00+00:00    24.979376388246145    50.05499629801029     None            None
        # 1973-05-07 12:32:00+00:00    24.927010515561776    50.194638625168594    None            None
        # ...

        min_max_anomaly_detector = MinMaxAnomalyDetector()
        evaluation_results = evaluate_anomaly_detector(min_max_anomaly_detector, humi_temp_df, synthetic=True)
        self.assertIsInstance(evaluation_results,dict)
        self.assertEqual(len(evaluation_results),2)
        self.assertIn('spike_uv',evaluation_results.keys())
        self.assertIn('false_positives',evaluation_results.keys())
        self.assertIsInstance(evaluation_results['spike_uv'],bool)
        self.assertIsInstance(evaluation_results['false_positives'],int)
        self.assertEqual(evaluation_results['false_positives'],2)
        # The detector does not see the downward spike in temperature as anomalous because the min temperature
        # value is 10.
        # The detector does not see the upward spike in humidity as anomalous because the max humidity
        # value is 70.
        # Evaluation_results:
        # { 'false_positives': 2
        #   'spike_uv':        False  
        # }

    def test_evaluate_anomaly_det_on_step_synth_timeseries(self):

        step_humi_temp_generator = SyntheticHumiTempTimeseriesGenerator()
        step_humi_temp_df = step_humi_temp_generator.generate(anomalies=['step_uv'],effects=[])
        # Generated DataFrame:
        # Timestamp                    temperature           humidity             anomaly_label    effect_label
        # ...
        # 1973-05-25 13:32:00+00:00   34.4037864008933       41.58990293095119    step_uv           None
        # ...
        min_max_anomaly_detector = MinMaxAnomalyDetector()
        evaluation_results = evaluate_anomaly_detector(min_max_anomaly_detector, step_humi_temp_df, synthetic=True)

        self.assertIsInstance(evaluation_results,dict)
        self.assertEqual(len(evaluation_results),2)
        self.assertIn('step_uv',evaluation_results.keys())
        self.assertIn('false_positives',evaluation_results.keys())
        self.assertIsInstance(evaluation_results['step_uv'],bool)
        self.assertIsInstance(evaluation_results['false_positives'],int)
        self.assertEqual(evaluation_results['false_positives'],1)
        # Evaluation results:
        # { 'false_positives': 1
        #   'step_uv':         True
        # }

    def test_evaluate_anomaly_det_on_synth_not_anomalous_timeseries(self):
        humi_temp_generator = SyntheticHumiTempTimeseriesGenerator()
        humi_temp_df = humi_temp_generator.generate(anomalies=[],effects=[])
        # Generated DataFrame:
        # Timestamp                    temperature           humidity             anomaly_label    effect_label
        # ...
        # 1973-05-07 11:02:00+00:00    24.761107302835810    50.63704719243785     None            None
        # 1973-05-07 11:17:00+00:00    24.868377982941322    50.350992045489804    None            None
        # 1973-05-07 11:32:00+00:00    24.944096137309916    50.14907696717356     None            None
        # ...
        min_max_anomaly_detector = MinMaxAnomalyDetector()
        evaluation_results = evaluate_anomaly_detector(min_max_anomaly_detector, humi_temp_df, synthetic=True)
        self.assertIsInstance(evaluation_results,dict)
        self.assertEqual(len(evaluation_results),1)
        self.assertIn('false_positives',evaluation_results.keys())
        self.assertIsInstance(evaluation_results['false_positives'],int)
        self.assertEqual(evaluation_results['false_positives'],2)
        # Evaluation results:
        # { 'false_positives': 2
        # }

