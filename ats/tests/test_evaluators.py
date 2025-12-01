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
        single_model_evaluation = { 'sample_1': {'anomalies_count': 3, 'anomalies_ratio': 1.5,
                                                    'false_positives_count': 1, 
                                                    'false_positives_ratio': 0.14},
                                    'sample_2': {'anomalies_count': 3, 'anomalies_ratio': 1.5,
                                                    'false_positives_count': 1, 
                                                    'false_positives_ratio': 0.14},
                                    'sample_3': {'anomalies_count': 3, 'anomalies_ratio': 1.5,
                                                    'false_positives_count': 1,
                                                    'false_positives_ratio': 0.14}
        }
        model_scores = _calculate_model_scores(single_model_evaluation)
        self.assertIsInstance(model_scores,dict)
        self.assertIn('anomalies_count',model_scores.keys())
        self.assertIn('anomalies_ratio',model_scores.keys())
        self.assertIn('false_positives_count',model_scores.keys())
        self.assertIn('false_positives_ratio',model_scores.keys())

        self.assertAlmostEqual(model_scores['anomalies_count'],9)
        self.assertAlmostEqual(model_scores['anomalies_ratio'],1.5)
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
        evaluation_results = evaluator.evaluate(models=models,granularity='data_point')
        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_count'],6)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_ratio'],7/8)
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
        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_count'],7)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_ratio'],25/48)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],5)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],31/126)

    def test_evaluate_series_granularity(self):
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
        # Evaluation_results:
        # detector_1: {'step_uv': 0.5, 'false_positives': 0.5}
        # detector_2: {'step_uv': 0.5, 'false_positives': 0.5}
        # detector_3: {'step_uv': 0.5, 'false_positives': 0.5}

        self.assertIsInstance(evaluation_results,dict)
        self.assertEqual(len(evaluation_results),3)
        self.assertEqual(len(evaluation_results['detector_1']),2)
        self.assertEqual(len(evaluation_results['detector_2']),2)
        self.assertEqual(len(evaluation_results['detector_3']),2)
        self.assertAlmostEqual(evaluation_results['detector_1']['step_uv'],0.5)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives'],0.5)

    def test_series_granularity_eval_with_non_detected_anomalies(self):
        effects = []
        series_generator = HumiTempTimeseriesGenerator()
        # series_1 will be a true anomaly for the minmax
        series_1 = series_generator.generate(include_effect_label=True, anomalies=['step_uv'],effects=effects)
        # series_2 will be a false positive for minmax (it sees always 2 anomalous data points for each variable)
        series_2 = series_generator.generate(include_effect_label=True, anomalies=['pattern_uv'],effects=effects)
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
        # Evaluation_results:
        # detector_1: {'step_uv': 0.5, 'pattern_uv': 0.5, 'false_positives': 0.0}
        # detector_2: {'step_uv': 0.5, 'pattern_uv': 0.5, 'false_positives': 0.0}
        # detector_3: {'step_uv': 0.5, 'pattern_uv': 0.5, 'false_positives': 0.0}
        self.assertIsInstance(evaluation_results,dict)
        self.assertEqual(len(evaluation_results),3)
        self.assertEqual(len(evaluation_results['detector_1']),3)
        self.assertEqual(len(evaluation_results['detector_2']),3)
        self.assertEqual(len(evaluation_results['detector_3']),3)
        self.assertAlmostEqual(evaluation_results['detector_1']['step_uv'],0.5)
        self.assertAlmostEqual(evaluation_results['detector_1']['pattern_uv'],0.5)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives'],0.0)

    def test_raised_error_evaluation_series_granularity(self):
        anomalies = ['step_uv','spike_uv']
        series_generator = HumiTempTimeseriesGenerator()
        series = series_generator.generate(include_effect_label=True, anomalies=anomalies)
        dataset = [series]
        minmax = MinMaxAnomalyDetector()
        evaluator = Evaluator(test_data=dataset)
        try:
            evaluation_result = evaluator.evaluate(models={'detector':minmax},granularity='series')
        except Exception as e:
            self.assertIsInstance(e,ValueError)

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
        self.assertIn('anomalies_count',evaluation_results['detector_1'].keys())
        self.assertIn('anomalies_ratio',evaluation_results['detector_1'].keys())
        self.assertIn('false_positives_count',evaluation_results['detector_1'].keys())
        self.assertIn('false_positives_ratio',evaluation_results['detector_1'].keys())

        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_count'],4)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_ratio'],4/6)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],0)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],0)

        dataset1 = [self.series2]
        evaluator1 = Evaluator(test_data=dataset1)
        evaluation_results = evaluator1.evaluate(models=models,granularity='variable')
        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_count'],3)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_ratio'],3/8)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],1)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],1/(7*2))

    def test_point_granularity_evaluation(self):
        dataset = [self.series1]
        evaluator = Evaluator(test_data=dataset)
        minmax1 = MinMaxAnomalyDetector()
        models={'detector_1': minmax1}
        evaluator = Evaluator(test_data=dataset)
        evaluation_results = evaluator.evaluate(models=models,granularity='data_point')
        self.assertIn('detector_1',evaluation_results.keys())
        self.assertIn('anomalies_count',evaluation_results['detector_1'].keys())
        self.assertIn('anomalies_ratio',evaluation_results['detector_1'].keys())
        self.assertIn('false_positives_count',evaluation_results['detector_1'].keys())
        self.assertIn('false_positives_ratio',evaluation_results['detector_1'].keys())

        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_count'],3)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_ratio'],3/3)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],0)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],0)

        dataset1 = [self.series2]
        evaluator1 = Evaluator(test_data=dataset1)
        evaluation_results = evaluator1.evaluate(models=models,granularity='data_point')
        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_count'],3)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_ratio'],3/4)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],1)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],1/7)

    def test_series_granularity_evaluation(self):
        dataset = [self.series1]
        evaluator = Evaluator(test_data=dataset)
        minmax1 = MinMaxAnomalyDetector()
        models={'detector_1': minmax1}
        evaluator = Evaluator(test_data=dataset)
        evaluation_results = evaluator.evaluate(models=models,granularity='series')
        self.assertIn('detector_1',evaluation_results.keys())
        self.assertIn('anomalies_count',evaluation_results['detector_1'].keys())
        self.assertIn('anomalies_ratio',evaluation_results['detector_1'].keys())
        self.assertIn('false_positives_count',evaluation_results['detector_1'].keys())
        self.assertIn('false_positives_ratio',evaluation_results['detector_1'].keys())

        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_count'],1)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_ratio'],1)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],0)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],0)

        dataset1 = [self.series3]
        evaluator1 = Evaluator(test_data=dataset1)
        evaluation_results = evaluator1.evaluate(models=models,granularity='series')
        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_count'],0)
        self.assertAlmostEqual(evaluation_results['detector_1']['anomalies_ratio'],0)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_count'],1)
        self.assertAlmostEqual(evaluation_results['detector_1']['false_positives_ratio'],1)

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
