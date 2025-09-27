from ..evaluators import evaluate_anomaly_detector
from ..anomaly_detectors import MinMaxAnomalyDetector
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
        #{'normal':      timestamp                   value_anomaly
        #                2025-06-10 15:00:00+00:00   False,
        #
        #                timestamp                   value_anomaly
        # 'anomaly_1':   2025-06-10 15:00:00+00:00   True,
        #
        #                timestamp                   value_anomaly
        # 'anomaly_2':   2025-06-10 14:00:00+00:00   True
        #}

        self.assertIsInstance(evaluation_results,dict)
        self.assertEqual(len(evaluation_results),3)
        self.assertIsInstance(evaluation_results['anomaly_1'],pd.DataFrame)
        self.assertIsInstance(evaluation_results['anomaly_2'],pd.DataFrame)
        self.assertIsInstance(evaluation_results['false positive'],bool)
        self.assertEqual(evaluation_results['anomaly_2'].loc['2025-06-10 16:00:00+00:00','value_anomaly'],True)
        self.assertEqual(evaluation_results['anomaly_1'].loc['2025-06-10 14:00:00+00:00','value_anomaly'],True)
        self.assertEqual(evaluation_results['false positive'],False)

    def test_evaluate_anomaly_det_on_spiked_synth_timeseries(self):

        spiked_humi_temp_generator = SyntheticHumiTempTimeseriesGenerator()
        humi_temp_df = spiked_humi_temp_generator.generate(anomalies=['spike_uv'],effects=[])
        # Generated DataFrame:
        # Timestamp                    temperature           humidity             anomaly_label    effect_label
        # ...
        # 1973-05-07 11:02:00+00:00    24.761107302835810    50.63704719243785    normal           bare
        # 1973-05-07 11:17:00+00:00    24.868377982941322    50.350992045489804   normal           bare
        # 1973-05-07 11:32:00+00:00    24.944096137309916    50.14907696717356    normal           bare
        # 1973-05-07 11:47:00+00:00    15.987937529180115    59.03216658885302    spike_uv         bare
        # 1973-05-07 12:02:00+00:00    24.999714422981285    50.000761538716574   normal           bare
        # 1973-05-07 12:17:00+00:00    24.979376388246145    50.05499629801029    normal           bare
        # 1973-05-07 12:32:00+00:00    24.927010515561776    50.194638625168594   normal           bare
        # ...
        max_temp_index = humi_temp_df['temperature'].idxmax()
        min_temp_index = humi_temp_df['temperature'].idxmin()
        max_humi_index = humi_temp_df['humidity'].idxmax()
        min_humi_index = humi_temp_df['humidity'].idxmin()
        label_detected_anomalous_temp1 = humi_temp_df.loc[max_temp_index,'anomaly_label'] if humi_temp_df.loc[max_temp_index,'anomaly_label'] != 'normal' else 'false positive'
        label_detected_anomalous_temp2 = humi_temp_df.loc[min_temp_index,'anomaly_label'] if humi_temp_df.loc[min_temp_index,'anomaly_label'] != 'normal' else 'false positive'
        label_detected_anomalous_humi1 = humi_temp_df.loc[max_humi_index,'anomaly_label'] if humi_temp_df.loc[max_humi_index,'anomaly_label'] != 'normal' else 'false positive'
        label_detected_anomalous_humi2 = humi_temp_df.loc[min_humi_index,'anomaly_label'] if humi_temp_df.loc[min_humi_index,'anomaly_label'] != 'normal' else 'false positive'

        min_max_anomaly_detector = MinMaxAnomalyDetector()
        evaluation_results = evaluate_anomaly_detector(min_max_anomaly_detector, humi_temp_df, synthetic=True)
        self.assertEqual(len(evaluation_results),2)
        self.assertIsInstance(evaluation_results[label_detected_anomalous_temp1],pd.DataFrame)
        self.assertIsInstance(evaluation_results[label_detected_anomalous_temp2],pd.DataFrame)
        self.assertIsInstance(evaluation_results[label_detected_anomalous_humi1],pd.DataFrame)
        self.assertIsInstance(evaluation_results[label_detected_anomalous_humi2],pd.DataFrame)

        # The detector does not see the downward spike in temperature as anomalous because the min temperature
        # value is 10.
        # The detector does not see the upward spike in humidity as anomalous because the max humidity
        # value is 70.
        # Evaluation_results:
        # { 'normal':                                 temperature_anomaly   humidity_anomaly
        #               1973-05-03 00:02:00+00:00     True                 False
        #               1973-05-11 23:47:00+00:00     False                True
        #               1973-05-13 12:02:00+00:00     True                 True,
        #
        # 'spike_uv':  False  
        # }

    def test_evaluate_anomaly_det_on_step_synth_timeseries(self):

        step_humi_temp_generator = SyntheticHumiTempTimeseriesGenerator()
        step_humi_temp_df = step_humi_temp_generator.generate(anomalies=['step_uv'],effects=[])
        # Generated DataFrame:
        # Timestamp                    temperature           humidity             anomaly_label    effect_label
        # ...
        # 1973-05-25 13:32:00+00:00   34.4037864008933       41.58990293095119    step_uv          bare
        # ...
        max_temp_index = step_humi_temp_df['temperature'].idxmax()
        min_temp_index = step_humi_temp_df['temperature'].idxmin()
        max_humi_index = step_humi_temp_df['humidity'].idxmax()
        min_humi_index = step_humi_temp_df['humidity'].idxmin()
        label_detected_anomalous_temp1 = step_humi_temp_df.loc[max_temp_index,'anomaly_label'] if step_humi_temp_df.loc[max_temp_index,'anomaly_label'] != 'normal' else 'false positive'
        label_detected_anomalous_temp2 = step_humi_temp_df.loc[min_temp_index,'anomaly_label'] if step_humi_temp_df.loc[min_temp_index,'anomaly_label'] != 'normal' else 'false positive'
        label_detected_anomalous_humi1 = step_humi_temp_df.loc[max_humi_index,'anomaly_label'] if step_humi_temp_df.loc[max_humi_index,'anomaly_label'] != 'normal' else 'false positive'
        label_detected_anomalous_humi2 = step_humi_temp_df.loc[min_humi_index,'anomaly_label'] if step_humi_temp_df.loc[min_humi_index,'anomaly_label'] != 'normal' else 'false positive'

        min_max_anomaly_detector = MinMaxAnomalyDetector()
        evaluation_results = evaluate_anomaly_detector(min_max_anomaly_detector, step_humi_temp_df, synthetic=True)
        self.assertEqual(len(evaluation_results),2)
        self.assertIsInstance(evaluation_results[label_detected_anomalous_temp1],pd.DataFrame)
        self.assertIsInstance(evaluation_results[label_detected_anomalous_temp2],pd.DataFrame)
        self.assertIsInstance(evaluation_results[label_detected_anomalous_humi1],pd.DataFrame)
        self.assertIsInstance(evaluation_results[label_detected_anomalous_humi2],pd.DataFrame)
        # Evaluation results:
        # { 'normal':                                 temperature_anomaly   humidity_anomaly
        #               1973-05-03 00:02:00+00:00     True                 True
        #
        # 'step_uv':                                  temperature_anomaly   humidity_anomaly
        #               1973-05-26 12:02:00+00:00     True                  True
        # }

    def test_evaluate_anomaly_det_on_synth_not_anomalous_timeseries(self):
        humi_temp_generator = SyntheticHumiTempTimeseriesGenerator()
        humi_temp_df = humi_temp_generator.generate(anomalies=[],effects=[])
        # Generated DataFrame:
        # Timestamp                    temperature           humidity             anomaly_label    effect_label
        # ...
        # 1973-05-07 11:02:00+00:00    24.761107302835810    50.63704719243785    normal           bare
        # 1973-05-07 11:17:00+00:00    24.868377982941322    50.350992045489804   normal           bare
        # 1973-05-07 11:32:00+00:00    24.944096137309916    50.14907696717356    normal           bare
        # ...
        max_temp_index = humi_temp_df['temperature'].idxmax()
        min_temp_index = humi_temp_df['temperature'].idxmin()
        max_humi_index = humi_temp_df['humidity'].idxmax()
        min_humi_index = humi_temp_df['humidity'].idxmin()
        label_detected_anomalous_temp1 = humi_temp_df.loc[max_temp_index,'anomaly_label'] if humi_temp_df.loc[max_temp_index,'anomaly_label'] != 'normal' else 'false positive'
        label_detected_anomalous_temp2 = humi_temp_df.loc[min_temp_index,'anomaly_label'] if humi_temp_df.loc[min_temp_index,'anomaly_label'] != 'normal' else 'false positive'
        label_detected_anomalous_humi1 = humi_temp_df.loc[max_humi_index,'anomaly_label'] if humi_temp_df.loc[max_humi_index,'anomaly_label'] != 'normal' else 'false positive'
        label_detected_anomalous_humi2 = humi_temp_df.loc[min_humi_index,'anomaly_label'] if humi_temp_df.loc[min_humi_index,'anomaly_label'] != 'normal' else 'false positive'

        min_max_anomaly_detector = MinMaxAnomalyDetector()
        evaluation_results = evaluate_anomaly_detector(min_max_anomaly_detector, humi_temp_df, synthetic=True)
        self.assertEqual(len(evaluation_results),1)
        self.assertIsInstance(evaluation_results[label_detected_anomalous_temp1],pd.DataFrame)
        self.assertIsInstance(evaluation_results[label_detected_anomalous_temp2],pd.DataFrame)
        self.assertIsInstance(evaluation_results[label_detected_anomalous_humi1],pd.DataFrame)
        self.assertIsInstance(evaluation_results[label_detected_anomalous_humi2],pd.DataFrame)
        # Evaluation results:
        # { 'normal':                              temperature_anomaly  humidity_anomaly
        #            1973-05-03 00:02:00+00:00     True                 True
        #            1973-05-03 12:02:00+00:00     True                 True
        # }

