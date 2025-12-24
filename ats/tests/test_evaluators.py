from ..evaluators import evaluate_anomaly_detector
from ..anomaly_detectors.naive import MinMaxAnomalyDetector
from ..timeseries_generators import HumiTempTimeseriesGenerator
from ..utils import generate_timeseries_df
from ..evaluators import _get_model_output
from ..evaluators import _format_for_anomaly_detector
from ..evaluators import _calculate_model_scores
from ..evaluators import Evaluator
from ..evaluators import _variable_granularity_evaluation
from ..evaluators import _point_granularity_evaluation
from ..evaluators import _series_granularity_evaluation
from ..evaluators import _get_breakdown_info
from ..anomaly_detectors.stat.periodic_average import PeriodicAverageAnomalyDetector
from ats.anomaly_detectors.stat.robust import NHARAnomalyDetector
from ..evaluators import _count_anomalous_events
from ..evaluators import _point_eval_with_events_strategy
from ats.dataset_generators import HumiTempDatasetGenerator

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

        self.series1 = generate_timeseries_df(entries=5, variables=2)
        self.series1['anomaly_label'] = [None, 'anomaly_2', 'anomaly_1', None, 'anomaly_1']
        # series1
        # timestamp                  value_1   value_2      anomaly_label                                               
        # 2025-06-10 14:00:00+00:00  0.000000  0.707107          None
        # 2025-06-10 15:00:00+00:00  0.841471  0.977061     anomaly_2
        # 2025-06-10 16:00:00+00:00  0.909297  0.348710     anomaly_1
        # 2025-06-10 17:00:00+00:00  0.141120 -0.600243          None
        # 2025-06-10 18:00:00+00:00 -0.756802 -0.997336     anomaly_1
        self.series2 = generate_timeseries_df(entries=7, variables=2)
        self.series2['anomaly_label'] = ['anomaly_1', 'anomaly_2', 'anomaly_1', None, 'anomaly_1', None, None]
        # series2
        # timestamp                  value_1   value_2      anomaly_label
        # 2025-06-10 14:00:00+00:00  0.000000  0.707107     anomaly_1
        # 2025-06-10 15:00:00+00:00  0.841471  0.977061     anomaly_2
        # 2025-06-10 16:00:00+00:00  0.909297  0.348710     anomaly_1
        # 2025-06-10 17:00:00+00:00  0.141120 -0.600243          None
        # 2025-06-10 18:00:00+00:00 -0.756802 -0.997336     anomaly_1
        # 2025-06-10 19:00:00+00:00 -0.958924 -0.477482          None
        # 2025-06-10 20:00:00+00:00 -0.279415  0.481366          None
        self.series3 = generate_timeseries_df(entries=3, variables=2)
        self.series3['anomaly_label'] = [None, None, None]

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
        timeseries_df = spiked_humi_temp_generator.generate(include_effect_label=True, anomalies=['spike_uv'],effects=[])
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
        try:
            evaluated_ts_df.drop(columns=['temperature_anomaly'],inplace=True)
            ev_results = evaluate_anomaly_detector(evaluated_ts_df,anomaly_labels)
        except NotImplementedError as error:
            self.assertIsInstance(error,NotImplementedError)

    def test_evaluate_anomaly_det_on_step_synth_timeseries(self):

        step_humi_temp_generator = HumiTempTimeseriesGenerator()
        timeseries_df = step_humi_temp_generator.generate(include_effect_label=True, anomalies=['step_uv'],effects=[])
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
        timeseries_df = humi_temp_generator.generate(include_effect_label=True, anomalies=[],effects=[])
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
        timeseries_df = spiked_humi_temp_generator.generate(include_effect_label=True, anomalies=['spike_uv'],effects=[])
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
        timeseries_df = step_humi_temp_generator.generate(include_effect_label=True, anomalies=['step_uv'],effects=[])

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
        timeseries_df = humi_temp_generator.generate(include_effect_label=True, anomalies=[],effects=[])

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
        humitemp_series1 = humi_temp_generator.generate(include_effect_label=True, anomalies=[],effects=[])
        humitemp_series2 = humi_temp_generator.generate(include_effect_label=True, anomalies=[],effects=['noise'])
        format_humitemp_series1,anomalies1 = _format_for_anomaly_detector(humitemp_series1,synthetic=True)
        format_humitemp_series2,anomalies2 = _format_for_anomaly_detector(humitemp_series2,synthetic=True)
        min_max = MinMaxAnomalyDetector()
        dataset = [format_humitemp_series1,format_humitemp_series2]
        flagged_dataset = _get_model_output(dataset,min_max)
        self.assertIsInstance(flagged_dataset,list)
        self.assertEqual(len(flagged_dataset),2)
        self.assertIn('temperature_anomaly',list(flagged_dataset[0].columns))
        self.assertIn('humidity_anomaly',list(flagged_dataset[0].columns))
        self.assertIn('temperature_anomaly',list(flagged_dataset[1].columns))
        self.assertIn('humidity_anomaly',list(flagged_dataset[1].columns))

    def test_calculate_model_scores(self):
        single_model_evaluation = { 'sample_1': {'true_positives_count': 3, 'true_positives_rate': 1.5,
                                                    'false_positives_count': 1, 
                                                    'false_positives_ratio': 0.14},
                                    'sample_2': {'true_positives_count': 3, 'true_positives_rate': 1.5,
                                                    'false_positives_count': 1, 
                                                    'false_positives_ratio': 0.14},
                                    'sample_3': {'true_positives_count': 3, 'true_positives_rate': 1.5,
                                                    'false_positives_count': 1,
                                                    'false_positives_ratio': 0.14}
        }
        model_scores = _calculate_model_scores(single_model_evaluation)
        self.assertIsInstance(model_scores,dict)
        self.assertIn('true_positives_count',model_scores.keys())
        self.assertIn('true_positives_rate',model_scores.keys())
        self.assertIn('false_positives_count',model_scores.keys())
        self.assertIn('false_positives_ratio',model_scores.keys())

        self.assertAlmostEqual(model_scores['true_positives_count'],9)
        self.assertAlmostEqual(model_scores['true_positives_rate'],1.5)
        self.assertAlmostEqual(model_scores['false_positives_count'],3)
        self.assertAlmostEqual(model_scores['false_positives_ratio'],0.14)

    def test_evaluate_point_granularity(self):
        dataset = [self.series1, self.series2, self.series3]
        minmax1 = MinMaxAnomalyDetector()
        minmax2 = MinMaxAnomalyDetector()
        minmax3 = MinMaxAnomalyDetector()
        models={'detector_1': minmax1,
                'detector_2': minmax2,
                'detector_3': minmax3
                }
        evaluator = Evaluator(test_data=dataset)
        evaluation_results = evaluator.evaluate(models=models,granularity='point')
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_count'],6)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_rate'],7/8)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],4)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],8/21)

    def test_evaluate_variable_granularity(self):
        dataset = [self.series1, self.series2, self.series3]
        minmax1 = MinMaxAnomalyDetector()
        minmax2 = MinMaxAnomalyDetector()
        minmax3 = MinMaxAnomalyDetector()
        models={'detector_1': minmax1,
                'detector_2': minmax2,
                'detector_3': minmax3
                }
        evaluator = Evaluator(test_data=dataset)
        evaluation_results = evaluator.evaluate(models=models,granularity='variable')
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_count'],7)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_rate'],25/48)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],5)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],31/126)

    def test_evaluate_series_granularity(self):
        dataset = [self.series1, self.series2, self.series3]
        minmax1 = MinMaxAnomalyDetector()
        minmax2 = MinMaxAnomalyDetector()
        minmax3 = MinMaxAnomalyDetector()
        models={'detector_1': minmax1,
                'detector_2': minmax2,
                'detector_3': minmax3
                }
        evaluator = Evaluator(test_data=dataset)
        evaluation_results = evaluator.evaluate(models=models,granularity='series')
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_count'],2)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_rate'],2/2)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],1)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],1/3)

    def test_copy_dataset(self):
        series_generator = HumiTempTimeseriesGenerator()
        series_1 = series_generator.generate(include_effect_label=True, effects=['noise'])
        series_2 = series_generator.generate(include_effect_label=True, effects=['noise'])
        dataset = [series_1,series_2]
        evaluator = Evaluator(test_data=dataset)
        minmax1 = MinMaxAnomalyDetector()
        minmax2 = MinMaxAnomalyDetector()
        dataset_copies = evaluator._copy_dataset(dataset,models=[minmax1,minmax2])
        self.assertIsInstance(dataset_copies,list)
        self.assertEqual(len(dataset_copies),2)
        self.assertIsInstance(dataset_copies[0],list)
        self.assertEqual(len(dataset_copies[0]),2)
        self.assertIsInstance(dataset_copies[1],list)
        self.assertEqual(len(dataset_copies[1]),2)

    def test_variable_granularity_evaluation(self):
        dataset = [self.series1]
        evaluator = Evaluator(test_data=dataset)
        minmax1 = MinMaxAnomalyDetector()
        models={'detector_1': minmax1}
        evaluator = Evaluator(test_data=dataset)
        evaluation_results = evaluator.evaluate(models=models,granularity='variable')
        self.assertIn('detector_1',evaluation_results.keys())
        self.assertIn('true_positives_count',evaluation_results['detector_1'].keys())
        self.assertIn('true_positives_rate',evaluation_results['detector_1'].keys())
        self.assertIn('false_positives_count',evaluation_results['detector_1'].keys())
        self.assertIn('false_positives_ratio',evaluation_results['detector_1'].keys())

        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_count'],4)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_rate'],4/6)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],0)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],0)

        dataset1 = [self.series2]
        evaluator1 = Evaluator(test_data=dataset1)
        evaluation_results = evaluator1.evaluate(models=models,granularity='variable')
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_count'],3)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_rate'],3/8)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],1)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],1/(7*2))

    def test_variable_granularity_evaluation_with_breakdown(self):
        formatted_series,anomaly_labels = _format_for_anomaly_detector(self.series1)
        minmax1 = MinMaxAnomalyDetector()
        flagged_series = _get_model_output([formatted_series],minmax1)
        evaluation_results = _variable_granularity_evaluation(flagged_series[0],anomaly_labels,breakdown=True)

        self.assertIn('true_positives_count',evaluation_results.keys())
        self.assertIn('true_positives_rate',evaluation_results.keys())
        self.assertIn('false_positives_count',evaluation_results.keys())
        self.assertIn('false_positives_ratio',evaluation_results.keys())

        self.assertIn('anomaly_1_true_positives_count',evaluation_results.keys())
        self.assertIn('anomaly_1_true_positives_rate',evaluation_results.keys())
        self.assertIn('anomaly_2_true_positives_count',evaluation_results.keys())
        self.assertIn('anomaly_2_true_positives_rate',evaluation_results.keys())

        self.assertAlmostEqual(evaluation_results['anomaly_1_true_positives_count'],3)
        self.assertAlmostEqual(evaluation_results['anomaly_1_true_positives_rate'],3/4)
        self.assertAlmostEqual(evaluation_results['anomaly_2_true_positives_count'],1)
        self.assertAlmostEqual(evaluation_results['anomaly_2_true_positives_rate'],1/2)

        formatted_series1,anomaly_labels1 = _format_for_anomaly_detector(self.series3)
        flagged_series1 = _get_model_output([formatted_series1],minmax1)
        evaluation_results1 = _variable_granularity_evaluation(flagged_series1[0],anomaly_labels1,breakdown=True)

        self.assertNotIn('anomaly_1_true_positives_count',evaluation_results1.keys())
        self.assertNotIn('anomaly_1_true_positives_rate',evaluation_results1.keys())
        self.assertNotIn('anomaly_2_true_positives_count',evaluation_results1.keys())
        self.assertNotIn('anomaly_2_true_positives_rate',evaluation_results1.keys())

    def test_point_granularity_evaluation(self):
        dataset = [self.series1]
        evaluator = Evaluator(test_data=dataset)
        minmax1 = MinMaxAnomalyDetector()
        models={'detector_1': minmax1}
        evaluator = Evaluator(test_data=dataset)
        evaluation_results = evaluator.evaluate(models=models,granularity='point')
        self.assertIn('detector_1',evaluation_results.keys())
        self.assertIn('true_positives_count',evaluation_results['detector_1'].keys())
        self.assertIn('true_positives_rate',evaluation_results['detector_1'].keys())
        self.assertIn('false_positives_count',evaluation_results['detector_1'].keys())
        self.assertIn('false_positives_ratio',evaluation_results['detector_1'].keys())

        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_count'],3)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_rate'],3/3)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],0)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],0)

        dataset1 = [self.series2]
        evaluator1 = Evaluator(test_data=dataset1)
        evaluation_results = evaluator1.evaluate(models=models,granularity='point')
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_count'],3)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_rate'],3/4)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],1)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],1/7)

    def test_point_granularity_evaluation_with_breakdown(self):
        formatted_series,anomaly_labels = _format_for_anomaly_detector(self.series1)
        minmax1 = MinMaxAnomalyDetector()
        flagged_series = _get_model_output([formatted_series],minmax1)
        evaluation_results = _point_granularity_evaluation(flagged_series[0],anomaly_labels,breakdown=True)

        self.assertIn('true_positives_count',evaluation_results.keys())
        self.assertIn('true_positives_rate',evaluation_results.keys())
        self.assertIn('false_positives_count',evaluation_results.keys())
        self.assertIn('false_positives_ratio',evaluation_results.keys())

        self.assertIn('anomaly_1_true_positives_count',evaluation_results.keys())
        self.assertIn('anomaly_1_true_positives_rate',evaluation_results.keys())
        self.assertIn('anomaly_2_true_positives_count',evaluation_results.keys())
        self.assertIn('anomaly_2_true_positives_rate',evaluation_results.keys())

        self.assertAlmostEqual(evaluation_results['anomaly_1_true_positives_count'],2)
        self.assertAlmostEqual(evaluation_results['anomaly_1_true_positives_rate'],2/2)
        self.assertAlmostEqual(evaluation_results['anomaly_2_true_positives_count'],1)
        self.assertAlmostEqual(evaluation_results['anomaly_2_true_positives_rate'],1/1)

    def test_series_granularity_evaluation(self):
        dataset = [self.series1]
        evaluator = Evaluator(test_data=dataset)
        minmax1 = MinMaxAnomalyDetector()
        models={'detector_1': minmax1}
        evaluator = Evaluator(test_data=dataset)
        evaluation_results = evaluator.evaluate(models=models,granularity='series')
        self.assertIn('detector_1',evaluation_results.keys())
        self.assertIn('true_positives_count',evaluation_results['detector_1'].keys())
        self.assertIn('true_positives_rate',evaluation_results['detector_1'].keys())
        self.assertIn('false_positives_count',evaluation_results['detector_1'].keys())
        self.assertIn('false_positives_ratio',evaluation_results['detector_1'].keys())

        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_count'],1)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_rate'],1)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],0)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],0)

        dataset1 = [self.series3]
        evaluator1 = Evaluator(test_data=dataset1)
        evaluation_results = evaluator1.evaluate(models=models,granularity='series')
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_count'],0)
        self.assertIsNone(evaluation_results['detector_1']['true_positives_rate'])
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],1)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],1)

    def test_series_granularity_evaluation_with_breakdown(self):
        series = generate_timeseries_df(entries=3, variables=2)
        series['anomaly_label'] = [None,None,'anomaly_1']
        formatted_series,anomaly_labels = _format_for_anomaly_detector(series)
        minmax1 = MinMaxAnomalyDetector()
        flagged_series = _get_model_output([formatted_series],minmax1)
        evaluation_results = _series_granularity_evaluation(flagged_series[0],anomaly_labels,breakdown=True)

        self.assertIn('true_positives_count',evaluation_results.keys())
        self.assertIn('true_positives_rate',evaluation_results.keys())
        self.assertIn('false_positives_count',evaluation_results.keys())
        self.assertIn('false_positives_ratio',evaluation_results.keys())
        self.assertIn('anomaly_1_true_positives_count',evaluation_results.keys())
        self.assertIn('anomaly_1_true_positives_rate',evaluation_results.keys())
        self.assertAlmostEqual(evaluation_results['anomaly_1_true_positives_count'],1)
        self.assertAlmostEqual(evaluation_results['anomaly_1_true_positives_rate'],1)

        formatted_series1,anomaly_labels1 = _format_for_anomaly_detector(self.series1)
        flagged_series1 = _get_model_output([formatted_series1],minmax1)
        try:
            evaluation_results = _point_granularity_evaluation(flagged_series1[0],anomaly_labels1,breakdown=True)
        except Exception as e:
            self.assertIsInstance(e,ValueError)

    def test_get_breakdown_info(self):
        single_model_evaluation = { 'sample_1': {'true_positives_count': 3, 'true_positives_rate': 1.5,
                                                    'false_positives_count': 1, 
                                                    'false_positives_ratio': 0.14,
                                                    'spike_true_positives_count': 1,
                                                    'spike_true_positives_rate': 0.5},
                                    'sample_2': {'true_positives_count': 3, 'true_positives_rate': 1.5,
                                                    'false_positives_count': 1, 
                                                    'false_positives_ratio': 0.14,
                                                    'spike_true_positives_count': 1,
                                                    'spike_true_positives_rate': 0.5,
                                                    'step_true_positives_count': 2,
                                                    'step_true_positives_rate': 2/3
                                                    },
                                    'sample_3': {'true_positives_count': 3, 'true_positives_rate': 1.5,
                                                    'false_positives_count': 1,
                                                    'false_positives_ratio': 0.14,
                                                    'step_true_positives_count': 3,
                                                    'step_true_positives_rate': 1,
                                                    'pattern_true_positives_count': 2,
                                                    'pattern_true_positives_rate': 0.5
                                                    }
        }
        breakdown = _get_breakdown_info(single_model_evaluation)
        self.assertIn('spike_true_positives_count',breakdown.keys())
        self.assertIn('spike_true_positives_rate',breakdown.keys())
        self.assertIn('step_true_positives_count',breakdown.keys())
        self.assertIn('step_true_positives_rate',breakdown.keys())
        self.assertIn('pattern_true_positives_count',breakdown.keys())
        self.assertIn('pattern_true_positives_rate',breakdown.keys())

        self.assertAlmostEqual(breakdown['spike_true_positives_count'],2)
        self.assertAlmostEqual(breakdown['spike_true_positives_rate'],1/2)
        self.assertAlmostEqual(breakdown['step_true_positives_count'],5)
        self.assertAlmostEqual(breakdown['step_true_positives_rate'],5/6)
        self.assertAlmostEqual(breakdown['pattern_true_positives_count'],2)
        self.assertAlmostEqual(breakdown['pattern_true_positives_rate'],0.5)

    def test_variable_granularity_eval_with_breakdown(self):
        dataset = [self.series1, self.series2, self.series3]
        minmax1 = MinMaxAnomalyDetector()
        minmax2 = MinMaxAnomalyDetector()
        minmax3 = MinMaxAnomalyDetector()
        models={'detector_1': minmax1,
                'detector_2': minmax2,
                'detector_3': minmax3
                }
        evaluator = Evaluator(test_data=dataset)
        evaluation_results = evaluator.evaluate(models=models,granularity='variable',breakdown=True)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_count'],7)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_rate'],25/48)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],5)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],31/126)

        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_1_true_positives_count'],5)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_1_true_positives_rate'],13/24)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_2_true_positives_count'],2)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_2_true_positives_rate'],1/2)

    def test_point_granularity_eval_with_breakdown(self):
        dataset = [self.series1, self.series2, self.series3]
        minmax1 = MinMaxAnomalyDetector()
        minmax2 = MinMaxAnomalyDetector()
        minmax3 = MinMaxAnomalyDetector()
        models={'detector_1': minmax1,
                'detector_2': minmax2,
                'detector_3': minmax3
                }
        evaluator = Evaluator(test_data=dataset)
        evaluation_results = evaluator.evaluate(models=models,granularity='point',breakdown=True)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_count'],6)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_rate'],7/8)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],4)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],8/21)

        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_1_true_positives_count'],4)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_1_true_positives_rate'],5/6)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_2_true_positives_count'],2)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_2_true_positives_rate'],1)

    def test_series_granularity_eval_with_breakdown(self):
        series_1 = generate_timeseries_df(entries=3, variables=2)
        series_1['anomaly_label'] = [None,None,'anomaly_1']
        series_2 = generate_timeseries_df(entries=3, variables=2)
        series_2['anomaly_label'] = ['anomaly_1',None,None]
        series_3 = generate_timeseries_df(entries=3, variables=2)
        series_3['anomaly_label'] = [None,'anomaly_2',None]
        dataset = [series_1, series_2, series_3]
        minmax1 = MinMaxAnomalyDetector()
        minmax2 = MinMaxAnomalyDetector()
        minmax3 = MinMaxAnomalyDetector()
        models={'detector_1': minmax1,
                'detector_2': minmax2,
                'detector_3': minmax3
                }
        evaluator = Evaluator(test_data=dataset)
        evaluation_results = evaluator.evaluate(models=models,granularity='series',breakdown=True)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_count'],3)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_rate'],1)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],0)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],0)

        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_1_true_positives_count'],2)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_1_true_positives_rate'],1)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_2_true_positives_count'],1)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_2_true_positives_rate'],1)

        try:
            dataset = [self.series1, self.series2, self.series3]
            evaluator = Evaluator(test_data=dataset)
            evaluation_results = evaluator.evaluate(models=models,granularity='series',breakdown=True)
        except Exception as e:
            self.assertIsInstance(e,ValueError)

    def test_double_evaluator(self):
        anomalies = ['step_uv']
        effects = []
        series_generator = HumiTempTimeseriesGenerator()
        # series_1 will be a true anomaly for the minmax
        series_1 = series_generator.generate(include_effect_label=True, anomalies=anomalies,effects=effects)
        # series_2 will be a false positive for minmax (it sees always 2 anomalous data points for each variable)
        series_2 = series_generator.generate(include_effect_label=True, anomalies=[],effects=effects)
        dataset = [series_1,series_2]
        evaluator = Evaluator(test_data=dataset)
        minmax1 = MinMaxAnomalyDetector()
        minmax2 = MinMaxAnomalyDetector()
        minmax3 = MinMaxAnomalyDetector()
        models={'detector_1': minmax1,
                'detector_2': minmax2,
                'detector_3': minmax3
                }
        evaluation_results = evaluator.evaluate(models=models,granularity='series')
        evaluation_results = evaluator.evaluate(models=models,granularity='series')

    def test_evaluate_with_autofit_model(self):

        anomalies = ['step_uv']
        effects = []
        series_generator = HumiTempTimeseriesGenerator()
        # series_1 will be a true anomaly for the minmax
        series_1 = series_generator.generate(include_effect_label=True, anomalies=anomalies,effects=effects)
        # series_2 will be a false positive for minmax (it sees always 2 anomalous data points for each variable)
        series_2 = series_generator.generate(include_effect_label=True, anomalies=[],effects=effects)
        dataset = [series_1,series_2]
        evaluator = Evaluator(test_data=dataset)
        models={'paverage': PeriodicAverageAnomalyDetector() }
        evaluation_results = evaluator.evaluate(models=models,granularity='point')

    def test_count_anomalous_events(self):
        humi_temp_generator = HumiTempTimeseriesGenerator()
        timeseries_df = humi_temp_generator.generate(include_effect_label=False, anomalies=['step_uv'])
        anomalous_events,events_by_type, event_time_slots = _count_anomalous_events(timeseries_df.loc[:,'anomaly_label'])
        self.assertEqual(anomalous_events,1)
        self.assertIsInstance(events_by_type,dict)
        self.assertEqual(events_by_type['step_uv'],1)

    def test_count_anomalous_events_with_point_anomaly(self):
        humi_temp_generator = HumiTempTimeseriesGenerator()
        timeseries_df = humi_temp_generator.generate(include_effect_label=False, anomalies=['spike_uv'])
        anomalous_events,events_by_type, event_time_slots = _count_anomalous_events(timeseries_df.loc[:,'anomaly_label'])
        self.assertEqual(anomalous_events,1)
        self.assertEqual(events_by_type['spike_uv'],1)

    def test_point_eval_with_events_strategy(self):
        # model output
        series = generate_timeseries_df(entries=6, variables=1)
        series['value_anomaly'] = [0,1,1,1,1,1]

        anomaly_labels = pd.Series([None, 'anomaly_1', 'anomaly_1', None, None,'anomaly_1'])
        anomaly_labels.index = series.index
        evaluation_result = _point_eval_with_events_strategy(series,anomaly_labels)
        self.assertAlmostEqual(evaluation_result['true_positives_count'],2)
        self.assertAlmostEqual(evaluation_result['true_positives_rate'],2/2)
        self.assertAlmostEqual(evaluation_result['false_positives_count'],1)
        self.assertAlmostEqual(evaluation_result['false_positives_ratio'],1/6)

    def test_point_eval_with_events_strategy_and_breakdown(self):
        # model output
        series = generate_timeseries_df(entries=6, variables=1)
        series['value_anomaly'] = [0,1,1,1,1,1]

        anomaly_labels = pd.Series([None, 'anomaly_1', 'anomaly_1', None, None,'anomaly_1'])
        anomaly_labels.index = series.index
        evaluation_result = _point_eval_with_events_strategy(series,anomaly_labels,breakdown=True)
        self.assertIn('anomaly_1_true_positives_count',evaluation_result.keys())
        self.assertAlmostEqual(evaluation_result['anomaly_1_true_positives_count'],2)
        self.assertAlmostEqual(evaluation_result['anomaly_1_true_positives_rate'],1)

    def test_eval_point_granularity_events_strategy(self):
        dataset = [self.series1, self.series2, self.series3]
        minmax1 = MinMaxAnomalyDetector()
        minmax2 = MinMaxAnomalyDetector()
        minmax3 = MinMaxAnomalyDetector()
        models={'detector_1': minmax1,
                'detector_2': minmax2,
                'detector_3': minmax3
                }
        evaluator = Evaluator(test_data=dataset)
        evaluation_results = evaluator.evaluate(models=models,granularity='point',strategy='events')
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_count'],6)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_rate'],7/8)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],2)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],10/(21*3))

    def test_eval_point_granularity_events_strategy_with_breakdown(self):
        dataset = [self.series1, self.series2, self.series3]
        minmax = MinMaxAnomalyDetector()
        models={'detector_1': minmax}
        evaluator = Evaluator(test_data=dataset)
        evaluation_results = evaluator.evaluate(models=models,granularity='point',strategy='events',breakdown=True)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_count'],6)
        self.assertAlmostEqual(evaluation_results['detector_1']['true_positives_rate'],7/8)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],2)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],10/(21*3))
        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_1_true_positives_count'],4)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_1_true_positives_rate'],5/6)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_2_true_positives_count'],2)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomaly_2_true_positives_rate'],1)

    def test_correct_counting_false_positives_with_events_strategy(self):
        effects = ['noise', 'clouds']
        anomalies = ['spike_mv', 'step_mv']
        generator = HumiTempDatasetGenerator(sampling_interval='60m')
        evaluation_dataset = generator.generate(n_series = 1, effects = effects, anomalies = anomalies,
                                                time_span = '90D', max_anomalies_per_series = 3, 
                                                anomalies_ratio = 1.0, auto_repeat_anomalies=True)
        models = {'minmax': MinMaxAnomalyDetector(), 
                  'nhar': NHARAnomalyDetector(),
                  'p_avg': PeriodicAverageAnomalyDetector()
                  }
        evaluator = Evaluator(test_data = evaluation_dataset)

        series = evaluation_dataset[0]
        anomalous_events_n, events_by_type_n, event_time_slots= _count_anomalous_events(series.loc[:,'anomaly_label'])
        evaluation_results = evaluator.evaluate(models=models,granularity='point',strategy='events',breakdown=False)

        for model in evaluation_results.keys():
            tp_n = evaluation_results[model]['true_positives_count']
            tp_rate = evaluation_results[model]['true_positives_rate']
            if tp_rate:
                self.assertAlmostEqual(anomalous_events_n, tp_n / tp_rate)

            fp_n = evaluation_results[model]['false_positives_count']
            fp_ratio = evaluation_results[model]['false_positives_ratio']
            if fp_ratio:
                self.assertAlmostEqual(len(series), fp_n / fp_ratio)

    def test_count_anomalous_events_on_synth_dataset(self):
        anomalies = ['step_mv','pattern_mv','spike_mv']
        generator = HumiTempDatasetGenerator(sampling_interval='60m')
        evaluation_dataset = generator.generate(n_series = 3, effects = ['noise'], anomalies = anomalies,
                                                time_span = '90D', max_anomalies_per_series = 3, 
                                                anomalies_ratio = 1.0, auto_repeat_anomalies=True)
        series_1 = evaluation_dataset[0]
        '''for timestamp in series_1.index:
            print(series_1.loc[timestamp,'anomaly_label'])'''
        # series_1
        # 1 step 
        # 0 pattern
        # 0 spike
        anomalous_events_n, events_by_type_n ,event_time_slots= _count_anomalous_events(series_1.loc[:,'anomaly_label'])
        self.assertIn('step_mv',events_by_type_n.keys())
        self.assertEqual(events_by_type_n['step_mv'],1)
        self.assertEqual(anomalous_events_n,1)

        series_2 = evaluation_dataset[1]
        '''for timestamp in series_2.index:
            print(series_2.loc[timestamp,'anomaly_label'])'''
        # series_2
        # 1 step 
        # 0 pattern
        # 0 spike
        anomalous_events_n_2, events_by_type_n_2,event_time_slots_2 = _count_anomalous_events(series_2.loc[:,'anomaly_label'])
        self.assertIn('step_mv',events_by_type_n_2.keys())
        self.assertEqual(events_by_type_n_2['step_mv'],1)
        self.assertEqual(anomalous_events_n_2,1)

        series_3 = evaluation_dataset[2]
        '''for timestamp in series_3.index:
            print(series_3.loc[timestamp,'anomaly_label'])'''
        # series_3
        # 1 step 
        # 1 pattern
        # 1 spike
        anomalous_events_n_3, events_by_type_n_3 , event_time_slots_3= _count_anomalous_events(series_3.loc[:,'anomaly_label'])
        self.assertIn('step_mv',events_by_type_n_3.keys())
        self.assertIn('pattern_mv',events_by_type_n_3.keys())
        self.assertEqual(events_by_type_n_3['step_mv'],1)
        self.assertEqual(events_by_type_n_3['pattern_mv'],1)
        self.assertEqual(events_by_type_n_3['spike_mv'],1)
        self.assertEqual(anomalous_events_n_3,3)

    def test_event_eval_on_p_avg(self):
        anomalies = ['step_mv']
        generator = HumiTempDatasetGenerator(sampling_interval='60m')
        evaluation_dataset = generator.generate(n_series = 1, effects = ['noise'], anomalies = anomalies,
                                                time_span = '90D', max_anomalies_per_series = 1, 
                                                anomalies_ratio = 1.0, auto_repeat_anomalies=True)
        series = evaluation_dataset[0]
        '''for timestamp in series.index:
            print(series.loc[timestamp,'anomaly_label'])'''
        # series
        # 1 step 
        # 0 pattern
        # 0 spike
        anomalous_events_n, events_by_type_n ,event_time_slots= _count_anomalous_events(series.loc[:,'anomaly_label'])
        self.assertIn('step_mv',events_by_type_n.keys())
        self.assertEqual(events_by_type_n['step_mv'],1)
        self.assertEqual(anomalous_events_n,1)

        model = PeriodicAverageAnomalyDetector()
        new_series = series.drop(columns=['anomaly_label'],inplace=False)
        p_avg_output = model.apply(new_series)

        anomalous_timestamps = []
        for timestamp in p_avg_output.index:
            is_anomalous = p_avg_output.filter(like='anomaly').loc[timestamp].any()
            anomaly_label = series.loc[timestamp,'anomaly_label']
            if anomaly_label is not None and is_anomalous:
                anomalous_timestamps.append(timestamp)

        start = anomalous_timestamps[0]
        sampling_interval = pd.Timedelta(minutes=60)
        consecutive_timestamp_n = 0
        for timestamp in anomalous_timestamps:
            are_consecutive = (timestamp - start) == sampling_interval
            if are_consecutive:
                consecutive_timestamp_n += 1
            start = timestamp

        detected_anomalies = len(anomalous_timestamps) - consecutive_timestamp_n
        evaluator = Evaluator(test_data = evaluation_dataset)
        evaluation_results = evaluator.evaluate(models={'p_avg':model},granularity='point',strategy='events',breakdown=False)
        self.assertEqual(evaluation_results['p_avg']['true_positives_count'],detected_anomalies)
        self.assertEqual(evaluation_results['p_avg']['true_positives_rate'],detected_anomalies/anomalous_events_n)

    def test_event_eval_on_nhar(self):
        anomalies = ['step_mv']
        generator = HumiTempDatasetGenerator(sampling_interval='60m')
        evaluation_dataset = generator.generate(n_series = 1, effects = ['noise'], anomalies = anomalies,
                                                time_span = '90D', max_anomalies_per_series = 1, 
                                                anomalies_ratio = 1.0, auto_repeat_anomalies=True)
        series = evaluation_dataset[0]
        '''for timestamp in series.index:
            print(series.loc[timestamp,'anomaly_label'])'''
        # series
        # 1 step 
        # 0 pattern
        # 0 spike
        anomalous_events_n, events_by_type_n , event_time_slots= _count_anomalous_events(series.loc[:,'anomaly_label'])
        self.assertIn('step_mv',events_by_type_n.keys())
        self.assertEqual(events_by_type_n['step_mv'],1)
        self.assertEqual(anomalous_events_n,1)

        model = PeriodicAverageAnomalyDetector()
        new_series = series.drop(columns=['anomaly_label'],inplace=False)
        p_avg_output = model.apply(new_series)

        anomalous_timestamps = []
        for timestamp in p_avg_output.index:
            is_anomalous = p_avg_output.filter(like='anomaly').loc[timestamp].any()
            anomaly_label = series.loc[timestamp,'anomaly_label']
            if anomaly_label is not None and is_anomalous:
                anomalous_timestamps.append(timestamp)

        start = anomalous_timestamps[0]
        sampling_interval = pd.Timedelta(minutes=60)
        consecutive_timestamp_n = 0
        for timestamp in anomalous_timestamps:
            are_consecutive = (timestamp - start) == sampling_interval
            if are_consecutive:
                consecutive_timestamp_n += 1
            start = timestamp

        detected_anomalies = len(anomalous_timestamps) - consecutive_timestamp_n
        evaluator = Evaluator(test_data = evaluation_dataset)
        evaluation_results = evaluator.evaluate(models={'nhar':model},granularity='point',strategy='events',breakdown=False)
        self.assertEqual(evaluation_results['nhar']['true_positives_count'],detected_anomalies)
        self.assertEqual(evaluation_results['nhar']['true_positives_rate'],detected_anomalies/anomalous_events_n)
