import unittest
import pandas as pd
import datetime as dt
import numpy as np
import random as rnd
from ..synthetic_data import SyntheticHumiTempTimeseriesGenerator
from ..synthetic_data import generate_time_boundaries
from ..synthetic_data import add_step_anomaly
from ..synthetic_data import add_anomalous_noise
from ..synthetic_data import add_pattern_anomaly
from ..synthetic_data import generate_synthetic_humitemp_timeseries
from ..synthetic_data import add_clouds_effects
from ..synthetic_data import add_spike_anomaly

# Setup logging
from .. import logger
logger.setup()

class TestSyntheticHumiTempTimeseriesGenerator(unittest.TestCase):

    def setUp(self):
        rnd.seed(123)
        np.random.seed(123)

    def test_defaults(self):
        default_generator = SyntheticHumiTempTimeseriesGenerator()
        default_timeseries_df = default_generator.generate()

        self.assertIsInstance(default_timeseries_df,pd.DataFrame)
        self.assertEqual(len(default_timeseries_df),2880)
        # Class attribute type control
        self.assertIsInstance(default_generator.observation_window,dt.timedelta)
        self.assertIsInstance(default_generator.sampling_interval,dt.timedelta)
        self.assertIsNone(default_generator.starting_year)
        self.assertIsNone(default_generator.starting_month)
        self.assertIsNone(default_generator.starting_day)
        self.assertIsNone(default_generator.starting_hour)
        self.assertIsInstance(default_generator.temperature,bool)
        self.assertIsInstance(default_generator.humidity,bool)

        anomaly_labels_and_counts = default_timeseries_df['anomaly_label'].value_counts(dropna=False)
        total_counts = 0

        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts

        self.assertEqual(total_counts,len(default_timeseries_df))
        self.assertEqual(default_timeseries_df.loc[10,'anomaly_label'],'spike_uv')

        for i in range(2160,2836):
            self.assertEqual(default_timeseries_df.loc[i,'anomaly_label'],'step_uv')

    def test_pattern_uv_timeseries_generator(self):
        pattern_uv_anomaly_generator = SyntheticHumiTempTimeseriesGenerator()
        pattern_uv_timeseries_df = pattern_uv_anomaly_generator.generate(anomalies=['pattern_uv'])

        self.assertEqual(len(pattern_uv_timeseries_df),2880)
        anomaly_labels_and_counts = pattern_uv_timeseries_df['anomaly_label'].value_counts(dropna=False)
        total_counts = 0

        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts

        self.assertEqual(total_counts,len(pattern_uv_timeseries_df))

        for i in range(960,1637):
            self.assertEqual(pattern_uv_timeseries_df.loc[i,'anomaly_label'],'pattern_uv')

    def test_noise_uv_timeseries_generator(self):
        noise_uv_anomaly_generator = SyntheticHumiTempTimeseriesGenerator()
        noise_uv_timeseries_df = noise_uv_anomaly_generator.generate(anomalies=['noise_uv'])

        self.assertEqual(len(noise_uv_timeseries_df),2880)

        anomaly_labels_and_counts = noise_uv_timeseries_df['anomaly_label'].value_counts(dropna=False)
        total_counts = 0

        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts

        self.assertEqual(total_counts,len(noise_uv_timeseries_df))

        for i in range(576,768):
            self.assertEqual(noise_uv_timeseries_df.loc[i,'anomaly_label'],'noise_uv')

    def test_spike_mv_timeseries_generator(self):
        spike_mv_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        spike_mv_timeseries_df = spike_mv_timeseries_generator.generate(anomalies=['spike_mv'])

        self.assertEqual(len(spike_mv_timeseries_df),2880)

        anomaly_labels_and_counts = spike_mv_timeseries_df['anomaly_label'].value_counts(dropna=False)
        total_counts = 0

        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts

        self.assertEqual(total_counts,len(spike_mv_timeseries_df))
        self.assertEqual(spike_mv_timeseries_df.loc[10,'anomaly_label'],'spike_mv')

    def test_step_mv_timeseries_generator(self):       
        step_mv_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        step_mv_timeseries_df = step_mv_timeseries_generator.generate(anomalies=['step_mv'])

        self.assertEqual(len(step_mv_timeseries_df),2880)             

        anomaly_labels_and_counts = step_mv_timeseries_df['anomaly_label'].value_counts(dropna=False)
        total_counts = 0

        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts

        self.assertEqual(total_counts,len(step_mv_timeseries_df))

        for i in range(2160,2836):
        	self.assertEqual(step_mv_timeseries_df.loc[i,'anomaly_label'],'step_mv')

    def test_pattern_mv_timeseries_generator(self):
        pattern_mv_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        pattern_mv_timeseries_df = pattern_mv_timeseries_generator.generate(anomalies=['pattern_mv'])

        self.assertEqual(len(pattern_mv_timeseries_df),2880)             

        anomaly_labels_and_counts = pattern_mv_timeseries_df['anomaly_label'].value_counts(dropna=False)
        total_counts = 0

        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts

        self.assertEqual(total_counts,len(pattern_mv_timeseries_df))

        for i in range(960,1637):
        	self.assertEqual(pattern_mv_timeseries_df.loc[i,'anomaly_label'],'pattern_mv')

    def test_noise_mv_timeseries_generator(self):
        noise_mv_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        noise_mv_timeseries_df = noise_mv_timeseries_generator.generate(anomalies=['noise_mv'])

        self.assertEqual(len(noise_mv_timeseries_df),2880)

        anomaly_labels_and_counts = noise_mv_timeseries_df['anomaly_label'].value_counts(dropna=False)
        total_counts = 0

        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts

        self.assertEqual(total_counts,len(noise_mv_timeseries_df))

        for i in range(576,768):
        	self.assertEqual(noise_mv_timeseries_df.loc[i,'anomaly_label'],'noise_mv')

    def test_clouds_mv_timeseries_generator(self):		       
        clouds_mv_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        clouds_mv_timeseries_df = clouds_mv_timeseries_generator.generate(anomalies=['clouds'],effects=['clouds'])

        self.assertEqual(len(clouds_mv_timeseries_df),2880)             

        anomaly_labels_and_counts = clouds_mv_timeseries_df['anomaly_label'].value_counts(dropna=False)
        total_counts = 0

        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts

        self.assertEqual(total_counts,len(clouds_mv_timeseries_df))

        for i in range(192,288):
        	self.assertEqual(clouds_mv_timeseries_df.loc[i,'anomaly_label'],'clouds')

    def test_all_uv_anomalies_timeseries_generator(self):
        all_uv_anomalies_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        all_uv_anomalies_timeseries_df = all_uv_anomalies_timeseries_generator.generate(anomalies=['spike_uv','step_uv','pattern_uv','noise_uv'])

        self.assertEqual(len(all_uv_anomalies_timeseries_df),2880)             

        anomaly_labels_and_counts = all_uv_anomalies_timeseries_df['anomaly_label'].value_counts(dropna=False)
        total_counts = 0

        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts

        self.assertEqual(total_counts,len(all_uv_anomalies_timeseries_df))
        self.assertEqual(all_uv_anomalies_timeseries_df.loc[10,'anomaly_label'],'spike_uv')

        for i in range(2160,2836):
            self.assertEqual(all_uv_anomalies_timeseries_df.loc[i,'anomaly_label'],'step_uv')

        for i in range(960,1637):
            self.assertEqual(all_uv_anomalies_timeseries_df.loc[i,'anomaly_label'],'pattern_uv')

        for i in range(576,768):
            self.assertEqual(all_uv_anomalies_timeseries_df.loc[i,'anomaly_label'],'noise_uv')

    def test_all_mv_anomalies_timeseries_generator(self):
        all_mv_anomalies_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        all_mv_anomalies_timeseries_df = all_mv_anomalies_timeseries_generator.generate(anomalies=['spike_mv','step_mv','pattern_mv','noise_mv','clouds'], effects=['clouds'])

        self.assertEqual(len(all_mv_anomalies_timeseries_df),2880)             

        anomaly_labels_and_counts = all_mv_anomalies_timeseries_df['anomaly_label'].value_counts(dropna=False)
        total_counts = 0

        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts

        self.assertEqual(total_counts,len(all_mv_anomalies_timeseries_df))
        # TODO: lo spike_mv viene sovrascritto dalle clouds
        #self.assertEqual(all_mv_anomalies_timeseries_df.loc[393,'anomaly_label'],'spike_mv')

        for i in range(2160,2836):
            self.assertEqual(all_mv_anomalies_timeseries_df.loc[i,'anomaly_label'],'step_mv')

        for i in range(960,1637):
            self.assertEqual(all_mv_anomalies_timeseries_df.loc[i,'anomaly_label'],'pattern_mv')

        for i in range(576,768):
            self.assertEqual(all_mv_anomalies_timeseries_df.loc[i,'anomaly_label'],'noise_mv')

        for i in range(96,192):
            self.assertEqual(all_mv_anomalies_timeseries_df.loc[i,'anomaly_label'],'clouds')

    def test_generate_time_boundaries(self):
        time_boundaries = generate_time_boundaries()
        self.assertIsInstance(time_boundaries,list)
        self.assertIsNotNone(time_boundaries[0])
        self.assertIsNotNone(time_boundaries[1])

    def test_add_step_anomaly(self):
        bare_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        bare_timeseries_df = bare_timeseries_generator.generate(effects=[],anomalies=[])
        step_uv_anomaly_timeseries_df = add_step_anomaly(bare_timeseries_df,mode='uv',inplace=False)
        step_mv_anomaly_timeseries_df = add_step_anomaly(bare_timeseries_df,mode='mv',inplace=False)

        for i in range(2217,2230):
            uv_temp_diff = step_uv_anomaly_timeseries_df.loc[i,'temperature'] - bare_timeseries_df.loc[i,'temperature']
            uv_humi_diff = step_uv_anomaly_timeseries_df.loc[i,'humidity'] - bare_timeseries_df.loc[i,'humidity']
            mv_temp_diff = step_mv_anomaly_timeseries_df.loc[i,'temperature'] - bare_timeseries_df.loc[i,'temperature']
            mv_humi_diff = step_mv_anomaly_timeseries_df.loc[i,'humidity'] - bare_timeseries_df.loc[i,'humidity']

            self.assertAlmostEqual(uv_temp_diff,10) 
            self.assertAlmostEqual(uv_humi_diff,-10)
            self.assertAlmostEqual(mv_temp_diff,10) 
            self.assertAlmostEqual(mv_humi_diff,10)

    def test_add_anomalous_noise(self):
        bare_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        bare_timeseries_df = bare_timeseries_generator.generate(effects=[],anomalies=[])
        uv_noise_anomaly_timeseries_df = add_anomalous_noise(bare_timeseries_df,inplace=False,mode='uv')
        mv_noise_anomaly_timeseries_df = add_anomalous_noise(bare_timeseries_df,inplace=False,mode='mv')

        for i in range(576,768):
            uv_temp_diff = uv_noise_anomaly_timeseries_df.loc[i,'temperature'] - bare_timeseries_df.loc[i,'temperature']
            uv_humi_diff = uv_noise_anomaly_timeseries_df.loc[i,'humidity'] - bare_timeseries_df.loc[i,'humidity']
            mv_temp_diff = mv_noise_anomaly_timeseries_df.loc[i,'temperature'] - bare_timeseries_df.loc[i,'temperature']
            mv_humi_diff = mv_noise_anomaly_timeseries_df.loc[i,'humidity'] - bare_timeseries_df.loc[i,'humidity']

            if i % 2 == 0:
                self.assertAlmostEqual(uv_temp_diff,3) 
                self.assertAlmostEqual(uv_humi_diff,3)
                self.assertAlmostEqual(mv_temp_diff,3)
                self.assertAlmostEqual(mv_humi_diff,0)

            else:
                self.assertAlmostEqual(uv_temp_diff,-3) 
                self.assertAlmostEqual(uv_humi_diff,-3)
                self.assertAlmostEqual(mv_temp_diff,-3)
                self.assertAlmostEqual(mv_humi_diff,0)

    def test_add_pattern_anomaly(self):
        bare_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        bare_timeseries_df = bare_timeseries_generator.generate(effects=[],anomalies=[])
        sampling_interval = dt.timedelta(minutes=15)
        uv_pattern_anomaly_timeseries_df = add_pattern_anomaly(bare_timeseries_df,sampling_interval,inplace=False,mode='uv')
        mv_pattern_anomaly_timeseries_df = add_pattern_anomaly(bare_timeseries_df,sampling_interval,inplace=False,mode='mv')

        # Uv pattern anomaly
        # Checking if the periodicity BEFORE the anomalous interval is the ordinary one (24h)
        before_anomaly_temp = uv_pattern_anomaly_timeseries_df.loc[5,'temperature']
        before_anomaly_temp_after_24h = uv_pattern_anomaly_timeseries_df.loc[101,'temperature']
        before_anomaly_humi = uv_pattern_anomaly_timeseries_df.loc[5,'humidity']
        before_anomaly_humi_after_24h = uv_pattern_anomaly_timeseries_df.loc[101,'humidity']
        self.assertAlmostEqual(before_anomaly_temp,before_anomaly_temp_after_24h)
        self.assertAlmostEqual(before_anomaly_humi,before_anomaly_humi_after_24h)

        # Checking if the anomalous periodicity is correct verifying if the temp(humi) is the same after 48h 
        anomalous_temp = uv_pattern_anomaly_timeseries_df.loc[1011,'temperature']
        anomalous_temp_after_48h = uv_pattern_anomaly_timeseries_df.loc[1203,'temperature']
        anomalous_humi = uv_pattern_anomaly_timeseries_df.loc[1011,'humidity']
        anomalous_humi_after_48h = uv_pattern_anomaly_timeseries_df.loc[1203,'humidity']
        self.assertAlmostEqual(anomalous_temp,anomalous_temp_after_48h)
        self.assertAlmostEqual(anomalous_humi,anomalous_humi_after_48h)

        # Checking if the periodicity AFTER the anomalous interval is the ordinary one (24h)
        after_anomaly_temp = uv_pattern_anomaly_timeseries_df.loc[1637,'temperature']
        after_anomaly_temp_after_24h = uv_pattern_anomaly_timeseries_df.loc[1733,'temperature']
        after_anomaly_humi = uv_pattern_anomaly_timeseries_df.loc[1637,'humidity']
        after_anomaly_humi_after_24h = uv_pattern_anomaly_timeseries_df.loc[1733,'humidity']
        self.assertAlmostEqual(after_anomaly_temp,after_anomaly_temp_after_24h)
        self.assertAlmostEqual(after_anomaly_humi,after_anomaly_humi_after_24h)

        # Mv pattern anomaly
        # Checking the anomalous periodicity to be correctly added verifying if the temp is the same after 48h
        # Also cheking humidity does not change in the case of mv anomaly
        anomalous_temp_mv = mv_pattern_anomaly_timeseries_df.loc[1011,'temperature']
        anomalous_temp__mv_after_48h = mv_pattern_anomaly_timeseries_df.loc[1203,'temperature']
        humi = mv_pattern_anomaly_timeseries_df.loc[1011,'humidity']
        humi_after_24h = mv_pattern_anomaly_timeseries_df.loc[1107,'humidity']
        self.assertAlmostEqual(anomalous_temp_mv,anomalous_temp__mv_after_48h)
        self.assertAlmostEqual(humi,humi_after_24h)

    def test_add_clouds_effects(self):
        bare_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        bare_timeseries_df = bare_timeseries_generator.generate(effects=[],anomalies=[])
        sampling_interval = dt.timedelta(minutes=15)
        clouds_effect_timeseries_df = add_clouds_effects(bare_timeseries_df,sampling_interval,inplace=False,mv_anomaly=True)

        delta_temp_first_half_of_the_day = bare_timeseries_df.loc[195,'temperature'] -clouds_effect_timeseries_df.loc[195,'temperature']

        espected_temp_difference1 = 0.01875 * bare_timeseries_df.loc[195,'temperature']
        self.assertAlmostEqual(delta_temp_first_half_of_the_day,espected_temp_difference1)

        delta_temp_second_half_of_the_day = bare_timeseries_df.loc[243,'temperature'] -clouds_effect_timeseries_df.loc[243,'temperature']

        espected_temp_difference2 = 0.28125 * bare_timeseries_df.loc[243,'temperature']
        self.assertAlmostEqual(delta_temp_second_half_of_the_day,espected_temp_difference2)

    def test_add_spike_anomaly(self):
        bare_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        bare_timeseries_df = bare_timeseries_generator.generate(effects=[],anomalies=[])
        uv_spiked_timeseries_df = add_spike_anomaly(bare_timeseries_df,inplace=False,mode='uv')
        mv_spiked_timeseries_df = add_spike_anomaly(bare_timeseries_df,inplace=False,mode='mv')

        uv_temp_diff = bare_timeseries_df.loc[10,'temperature'] - uv_spiked_timeseries_df.loc[10,'temperature']
        uv_humi_diff = uv_spiked_timeseries_df.loc[10,'humidity'] - bare_timeseries_df.loc[10,'humidity']
        self.assertAlmostEqual(uv_temp_diff,5)
        self.assertAlmostEqual(uv_humi_diff,5)

        mv_temp_diff = bare_timeseries_df.loc[10,'temperature'] - mv_spiked_timeseries_df.loc[10,'temperature']
        mv_humi_diff = mv_spiked_timeseries_df.loc[10,'humidity'] - bare_timeseries_df.loc[10,'humidity']
        self.assertAlmostEqual(mv_temp_diff,5)
        self.assertAlmostEqual(mv_humi_diff,5)

        for i in range(len(bare_timeseries_df)):

            if uv_spiked_timeseries_df.loc[i,'anomaly_label'] != 'spike_uv':
                self.assertAlmostEqual(bare_timeseries_df.loc[i,'humidity'],uv_spiked_timeseries_df.loc[i,'humidity'])
        # TODO: a way for knowing the intensity of the espected spike

    def test_default_effect_label(self):
        bare_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        bare_timeseries_df = bare_timeseries_generator.generate(effects=[],anomalies=[])
        self.assertIsNone(bare_timeseries_df.loc[10,'effect_label'])

    def test_normal_timeseries_label(self):
        timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        timeseries_df = timeseries_generator.generate(effects=[],anomalies=[])
        self.assertIsNone(timeseries_df.loc[10,'anomaly_label'])

