from ..evaluators import evaluate_anomaly_detector
from ..anomaly_detectors.naive import MinMaxAnomalyDetector
from ..timeseries_generators import HumiTempTimeseriesGenerator
from ..utils import generate_timeseries_df
from ..evaluators import get_model_output
from ..evaluators import _format_for_anomaly_detector
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
        f_timeseries_df,anomaly_labels = _format_for_anomaly_detector(timeseries_df)
        evaluated_ts_df = min_max_anomaly_detector.apply(f_timeseries_df)
        ev_details = evaluate_anomaly_detector(evaluated_ts_df,anomaly_labels)
        # Evaluation_results:
        #{ 'false_positives': 0,
        #  'anomaly_1':       True,
        #  'anomaly_2':       True
        #}

        self.assertIsInstance(ev_details,dict)
        self.assertEqual(len(ev_details),3)
        self.assertIn('anomaly_1',ev_details.keys())
        self.assertIn('anomaly_2',ev_details.keys())
        self.assertIn('false_positives',ev_details.keys())
        self.assertIsInstance(ev_details['anomaly_1'],bool)
        self.assertIsInstance(ev_details['anomaly_2'],bool)
        self.assertIsInstance(ev_details['false_positives'],int)
        self.assertEqual(ev_details['anomaly_2'],True)
        self.assertEqual(ev_details['anomaly_1'],True)
        self.assertEqual(ev_details['false_positives'],0)

    def test_evaluate_anomaly_det_on_spiked_synth_timeseries(self):

        spiked_humi_temp_generator = HumiTempTimeseriesGenerator()
        timeseries_df = spiked_humi_temp_generator.generate(anomalies=['spike_uv'],effects=[])
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
        f_timeseries_df,anomaly_labels = _format_for_anomaly_detector(timeseries_df,synthetic=True)
        evaluated_ts_df = min_max_anomaly_detector.apply(f_timeseries_df)
        ev_details = evaluate_anomaly_detector(evaluated_ts_df,anomaly_labels)
        self.assertIsInstance(ev_details,dict)
        self.assertEqual(len(ev_details),2)
        self.assertIn('spike_uv',ev_details.keys())
        self.assertIn('false_positives',ev_details.keys())
        self.assertIsInstance(ev_details['spike_uv'],bool)
        self.assertIsInstance(ev_details['false_positives'],int)
        self.assertEqual(ev_details['false_positives'],4)
        # The detector does not see the downward spike in temperature as anomalous because the min temperature
        # value is 10.
        # The detector does not see the upward spike in humidity as anomalous because the max humidity
        # value is 70.
        # Evaluation_results:
        # { 'false_positives': 4
        #   'spike_uv':        False  
        # }

    def test_evaluate_anomaly_det_on_step_synth_timeseries(self):

        step_humi_temp_generator = HumiTempTimeseriesGenerator()
        timeseries_df = step_humi_temp_generator.generate(anomalies=['step_uv'],effects=[])
        # Generated DataFrame:
        # Timestamp                    temperature           humidity             anomaly_label    effect_label
        # ...
        # 1973-05-25 13:32:00+00:00   34.4037864008933       41.58990293095119    step_uv           None
        # ...
        min_max_anomaly_detector = MinMaxAnomalyDetector()
        f_timeseries_df,anomaly_labels = _format_for_anomaly_detector(timeseries_df,synthetic=True)
        evaluated_ts_df = min_max_anomaly_detector.apply(f_timeseries_df)
        ev_details = evaluate_anomaly_detector(evaluated_ts_df,anomaly_labels)

        self.assertIsInstance(ev_details,dict)
        self.assertEqual(len(ev_details),2)
        self.assertIn('step_uv',ev_details.keys())
        self.assertIn('false_positives',ev_details.keys())
        self.assertIsInstance(ev_details['step_uv'],bool)
        self.assertIsInstance(ev_details['false_positives'],int)
        self.assertEqual(ev_details['false_positives'],2)
        # Evaluation results:
        # { 'false_positives': 2
        #   'step_uv':         True
        # }

    def test_evaluate_anomaly_det_on_synth_not_anomalous_timeseries(self):
        humi_temp_generator = HumiTempTimeseriesGenerator()
        timeseries_df = humi_temp_generator.generate(anomalies=[],effects=[])
        # Generated DataFrame:
        # Timestamp                    temperature           humidity             anomaly_label    effect_label
        # ...
        # 1973-05-07 11:02:00+00:00    24.761107302835810    50.63704719243785     None            None
        # 1973-05-07 11:17:00+00:00    24.868377982941322    50.350992045489804    None            None
        # 1973-05-07 11:32:00+00:00    24.944096137309916    50.14907696717356     None            None
        # ...
        min_max_anomaly_detector = MinMaxAnomalyDetector()
        f_timeseries_df,anomaly_labels = _format_for_anomaly_detector(timeseries_df,synthetic=True)
        evaluated_ts_df = min_max_anomaly_detector.apply(f_timeseries_df)
        ev_details = evaluate_anomaly_detector(evaluated_ts_df,anomaly_labels)

        self.assertIsInstance(ev_details,dict)
        self.assertEqual(len(ev_details),1)
        self.assertIn('false_positives',ev_details.keys())
        self.assertIsInstance(ev_details['false_positives'],int)
        self.assertEqual(ev_details['false_positives'],4)
        # Evaluation results:
        # { 'false_positives': 4
        # }

    def test_evaluation_details(self):
        min_max_anomaly_detector = MinMaxAnomalyDetector()
        timeseries_df = generate_timeseries_df(entries=4, variables=1)
        timeseries_df['anomaly_label'] = ['anomaly_1', None,'anomaly_2', None]
        f_timeseries_df,anomaly_labels = _format_for_anomaly_detector(timeseries_df)
        evaluated_ts_df = min_max_anomaly_detector.apply(f_timeseries_df)
        ev_results,ev_details = evaluate_anomaly_detector(evaluated_ts_df,anomaly_labels,details=True)

        self.assertIsInstance(ev_details,dict)
        self.assertEqual(len(ev_details),2)
        self.assertIn('anomaly_1',ev_details.keys())
        self.assertIn('anomaly_2',ev_details.keys())
        self.assertEqual(len(ev_details['anomaly_1']),1)
        self.assertEqual(len(ev_details['anomaly_2']),1)
        self.assertEqual(ev_details['anomaly_1'][pd.Timestamp('2025-06-10 14:00:00+00:00')]['value_anomaly'],1)
        self.assertEqual(ev_details['anomaly_2'][pd.Timestamp('2025-06-10 16:00:00+00:00')]['value_anomaly'],1)
        # Evaluation_details
        # anomaly_1: {Timestamp('2025-06-10 14:00:00+0000', tz='UTC'): {'value_anomaly': True}}
        # anomaly_2: {Timestamp('2025-06-10 16:00:00+0000', tz='UTC'): {'value_anomaly': True}}

    def test_evaluation_details_on_synth_spiked_timeseries(self):
        spiked_humi_temp_generator = HumiTempTimeseriesGenerator()
        timeseries_df = spiked_humi_temp_generator.generate(anomalies=['spike_uv'],effects=[])
        min_max_anomaly_detector = MinMaxAnomalyDetector()
        f_timeseries_df,anomaly_labels = _format_for_anomaly_detector(timeseries_df,synthetic=True)
        evaluated_ts_df = min_max_anomaly_detector.apply(f_timeseries_df)
        ev_results,ev_details = evaluate_anomaly_detector(evaluated_ts_df,anomaly_labels,details=True)
        self.assertIsInstance(ev_details,dict)
        self.assertEqual(len(ev_details),1)
        self.assertIsInstance(ev_details['false_positives'],dict)
        self.assertEqual(len(ev_details['false_positives']),2)
        self.assertIn('false_positives',ev_details.keys())
        self.assertNotIn('spike_uv',ev_details.keys())
        # Evaluation_details
        # false_positives: {Timestamp('1973-05-03 00:02:00+0000', tz='UTC'): {'temperature_anomaly': True, 'humidity_anomaly': True}, Timestamp('1973-05-03 12:02:00+0000', tz='UTC'): {'temperature_anomaly': True, 'humidity_anomaly': True}}

    def test_evaluation_details_on_synth_step_timeseries(self):
        step_humi_temp_generator = HumiTempTimeseriesGenerator()
        timeseries_df = step_humi_temp_generator.generate(anomalies=['step_uv'],effects=[])

        min_max_anomaly_detector = MinMaxAnomalyDetector()
        f_timeseries_df,anomaly_labels = _format_for_anomaly_detector(timeseries_df,synthetic=True)
        evaluated_ts_df = min_max_anomaly_detector.apply(f_timeseries_df)
        ev_results,ev_details = evaluate_anomaly_detector(evaluated_ts_df,anomaly_labels,details=True)

        self.assertIsInstance(ev_details,dict)
        self.assertEqual(len(ev_details),2)
        self.assertIn('step_uv',ev_details.keys())
        self.assertIn('false_positives',ev_details.keys())
        self.assertIsInstance(ev_details['step_uv'],dict)
        self.assertIsInstance(ev_details['false_positives'],dict)
        self.assertEqual(len(ev_details['step_uv']),1)
        self.assertEqual(len(ev_details['false_positives']),1)
        # Evaluation_details
        # step_uv: {Timestamp('1973-05-26 12:02:00+0000', tz='UTC'): {'temperature_anomaly': True, 'humidity_anomaly': True}}
        # false_positives: {Timestamp('1973-05-03 00:02:00+0000', tz='UTC'): {'temperature_anomaly': True, 'humidity_anomaly': True}}

    def test_evaluation_details_on_synth_not_anomalous_timeseries(self):
        humi_temp_generator = HumiTempTimeseriesGenerator()
        timeseries_df = humi_temp_generator.generate(anomalies=[],effects=[])

        min_max_anomaly_detector = MinMaxAnomalyDetector()
        f_timeseries_df,anomaly_labels = _format_for_anomaly_detector(timeseries_df,synthetic=True)
        evaluated_ts_df = min_max_anomaly_detector.apply(f_timeseries_df)
        ev_results,ev_details = evaluate_anomaly_detector(evaluated_ts_df,anomaly_labels,details=True)

        self.assertIsInstance(ev_details,dict)
        self.assertEqual(len(ev_details),1)
        self.assertIn('false_positives',ev_details.keys())
        self.assertIsInstance(ev_details['false_positives'],dict)
        self.assertEqual(len(ev_details['false_positives']),2)
        # Evaluation_details
        #false_positives: {Timestamp('1973-05-03 00:02:00+0000', tz='UTC'): {'temperature_anomaly': True, 'humidity_anomaly': True}, Timestamp('1973-05-03 12:02:00+0000', tz='UTC'): {'temperature_anomaly': True, 'humidity_anomaly': True}}

    def test_get_model_output(self):
        humi_temp_generator = HumiTempTimeseriesGenerator()
        humitemp_series1 = humi_temp_generator.generate(anomalies=[],effects=[])
        humitemp_series2 = humi_temp_generator.generate(anomalies=[],effects=['noise'])
        format_humitemp_series1,anomalies1 = _format_for_anomaly_detector(humitemp_series1,synthetic=True)
        format_humitemp_series2,anomalies2 = _format_for_anomaly_detector(humitemp_series2,synthetic=True)
        min_max = MinMaxAnomalyDetector()
        dataset = [format_humitemp_series1,format_humitemp_series2]
        flagged_dataset = get_model_output(dataset,min_max)
        self.assertIsInstance(flagged_dataset,list)
        self.assertEqual(len(flagged_dataset),2)
        self.assertIn('temperature_anomaly',list(flagged_dataset[0].columns))
        self.assertIn('humidity_anomaly',list(flagged_dataset[0].columns))
        self.assertIn('temperature_anomaly',list(flagged_dataset[1].columns))
        self.assertIn('humidity_anomaly',list(flagged_dataset[1].columns))
