import unittest
import pandas as pd
import datetime as dt
import numpy as np
import random as rnd

from ..synthetic_data import SyntheticHumiTempTimeseriesGenerator
from ..synthetic_data import generate_time_boundaries

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
        
        #class attribute type control
        self.assertIsInstance(default_generator.observation_window,dt.timedelta)
        self.assertIsInstance(default_generator.sampling_interval,dt.timedelta)
        self.assertIsNone(default_generator.starting_year)
        self.assertIsNone(default_generator.starting_month)
        self.assertIsNone(default_generator.starting_day)
        self.assertIsNone(default_generator.starting_hour)
        self.assertIsInstance(default_generator.temperature,bool)
        self.assertIsInstance(default_generator.humidity,bool)
             
        
        anomaly_labels_and_counts = default_timeseries_df['anomaly_label'].value_counts()
        total_counts = 0
        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts
        
        self.assertEqual(total_counts,len(default_timeseries_df))
        
        #for i in range(len(default_timeseries_df)):
            #print('{}: {}'.format(i,default_timeseries_df.loc[i,'anomaly_label']))
            
        self.assertEqual(default_timeseries_df.loc[562,'anomaly_label'],'spike_uv')
        for i in range(2160,2836):
            self.assertEqual(default_timeseries_df.loc[i,'anomaly_label'],'step_uv')
            
            
           
    def test_pattern_uv_timeseries_generator(self):
        
        pattern_uv_anomaly_generator = SyntheticHumiTempTimeseriesGenerator()
        pattern_uv_timeseries_df = pattern_uv_anomaly_generator.generate(anomalies=['pattern_uv'])
        
    
        self.assertEqual(len(pattern_uv_timeseries_df),2880)
             
     
        anomaly_labels_and_counts = pattern_uv_timeseries_df['anomaly_label'].value_counts()
        total_counts = 0
        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts
        
        self.assertEqual(total_counts,len(pattern_uv_timeseries_df))
        
        #for i in range(len(pattern_uv_timeseries_df)):
            #print('{}: {}'.format(i,pattern_uv_timeseries_df.loc[i,'anomaly_label']))
            
        for i in range(960,1637):
            self.assertEqual(pattern_uv_timeseries_df.loc[i,'anomaly_label'],'pattern_uv')
        


    def test_noise_uv_timeseries_generator(self):
       
        noise_uv_anomaly_generator = SyntheticHumiTempTimeseriesGenerator()
        noise_uv_timeseries_df = noise_uv_anomaly_generator.generate(anomalies=['noise_uv'])
        
    
        self.assertEqual(len(noise_uv_timeseries_df),2880)
             
     
        anomaly_labels_and_counts = noise_uv_timeseries_df['anomaly_label'].value_counts()
        total_counts = 0
        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts
        
        self.assertEqual(total_counts,len(noise_uv_timeseries_df))
        
        #for i in range(len(noise_uv_timeseries_df)):
            #print('{}: {}'.format(i,noise_uv_timeseries_df.loc[i,'anomaly_label']))
            
        for i in range(576,768):
            self.assertEqual(noise_uv_timeseries_df.loc[i,'anomaly_label'],'noise_uv')


#MV
    def test_spike_mv_timeseries_generator(self):
       
		       
        spike_mv_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        spike_mv_timeseries_df = spike_mv_timeseries_generator.generate(anomalies=['spike_mv'])
        
    
        self.assertEqual(len(spike_mv_timeseries_df),2880)
             
     
        anomaly_labels_and_counts = spike_mv_timeseries_df['anomaly_label'].value_counts()
        total_counts = 0
        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts
        
        self.assertEqual(total_counts,len(spike_mv_timeseries_df))
        
        #for i in range(len(spike_mv_timeseries_df)):
            #print('{}: {}'.format(i,spike_mv_timeseries_df.loc[i,'anomaly_label']))
            
        
        self.assertEqual(spike_mv_timeseries_df.loc[562,'anomaly_label'],'spike_mv')
        


    def test_step_mv_timeseries_generator(self):
       
		       
        step_mv_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        step_mv_timeseries_df = step_mv_timeseries_generator.generate(anomalies=['step_mv'])
        
    
        self.assertEqual(len(step_mv_timeseries_df),2880)
             
     
        anomaly_labels_and_counts = step_mv_timeseries_df['anomaly_label'].value_counts()
        total_counts = 0
        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts
        
        self.assertEqual(total_counts,len(step_mv_timeseries_df))
        
        #for i in range(len(step_mv_timeseries_df)):
         #   print('{}: {}'.format(i,step_mv_timeseries_df.loc[i,'anomaly_label']))
            
        for i in range(2160,2836):
        	self.assertEqual(step_mv_timeseries_df.loc[i,'anomaly_label'],'step_mv')



    def test_pattern_mv_timeseries_generator(self):
       
		       
        pattern_mv_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        pattern_mv_timeseries_df = pattern_mv_timeseries_generator.generate(anomalies=['pattern_mv'])
        
    
        self.assertEqual(len(pattern_mv_timeseries_df),2880)
             
     
        anomaly_labels_and_counts = pattern_mv_timeseries_df['anomaly_label'].value_counts()
        total_counts = 0
        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts
        
        self.assertEqual(total_counts,len(pattern_mv_timeseries_df))
        
        #for i in range(len(pattern_mv_timeseries_df)):
         #   print('{}: {}'.format(i,pattern_mv_timeseries_df.loc[i,'anomaly_label']))
            
        for i in range(960,1637):
        	self.assertEqual(pattern_mv_timeseries_df.loc[i,'anomaly_label'],'pattern_mv')



    def test_noise_mv_timeseries_generator(self):
       
		       
        noise_mv_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        noise_mv_timeseries_df = noise_mv_timeseries_generator.generate(anomalies=['noise_mv'])
        
    
        self.assertEqual(len(noise_mv_timeseries_df),2880)
             
     
        anomaly_labels_and_counts = noise_mv_timeseries_df['anomaly_label'].value_counts()
        total_counts = 0
        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts
        
        self.assertEqual(total_counts,len(noise_mv_timeseries_df))
        
        ##for i in range(len(noise_mv_timeseries_df)):
          #  print('{}: {}'.format(i,noise_mv_timeseries_df.loc[i,'anomaly_label']))
            
        for i in range(576,768):
        	self.assertEqual(noise_mv_timeseries_df.loc[i,'anomaly_label'],'noise_mv')



    def test_clouds_mv_timeseries_generator(self):
       
		       
        clouds_mv_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        clouds_mv_timeseries_df = clouds_mv_timeseries_generator.generate(anomalies=['clouds'],effects=['clouds'])
        
    
        self.assertEqual(len(clouds_mv_timeseries_df),2880)
             
     
        anomaly_labels_and_counts = clouds_mv_timeseries_df['anomaly_label'].value_counts()
        total_counts = 0
        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts
        
        self.assertEqual(total_counts,len(clouds_mv_timeseries_df))
        
        #for i in range(len(clouds_mv_timeseries_df)):
            #print('{}: {}'.format(i,clouds_mv_timeseries_df.loc[i,'anomaly_label']))
            
        for i in range(192,288):
        	self.assertEqual(clouds_mv_timeseries_df.loc[i,'anomaly_label'],'clouds')



    def test_all_uv_anomalies_timeseries_generator(self):
       
		       
        all_uv_anomalies_timeseries_generator = SyntheticHumiTempTimeseriesGenerator()
        all_uv_anomalies_timeseries_df = all_uv_anomalies_timeseries_generator.generate(anomalies=['spike_uv','step_uv','pattern_uv','noise_uv'])
        
    
        self.assertEqual(len(all_uv_anomalies_timeseries_df),2880)
             
     
        anomaly_labels_and_counts = all_uv_anomalies_timeseries_df['anomaly_label'].value_counts()
        total_counts = 0
        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts
        
        self.assertEqual(total_counts,len(all_uv_anomalies_timeseries_df))
        
        #for i in range(len(all_uv_anomalies_timeseries_df)):
            #print('{}: {}'.format(i,all_uv_anomalies_timeseries_df.loc[i,'anomaly_label']))
        
        self.assertEqual(all_uv_anomalies_timeseries_df.loc[562,'anomaly_label'],'spike_uv')
        
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
             
     
        anomaly_labels_and_counts = all_mv_anomalies_timeseries_df['anomaly_label'].value_counts()
        total_counts = 0
        for label_counts in anomaly_labels_and_counts:
           total_counts += label_counts
        
        self.assertEqual(total_counts,len(all_mv_anomalies_timeseries_df))
        #for i in range(len(all_mv_anomalies_timeseries_df)):
            #print('{}: {}'.format(i,all_mv_anomalies_timeseries_df.loc[i,'anomaly_label']))
        
        # TODO: lo spike_mv viene sovrascritto dalle clouds
        #self.assertEqual(all_mv_anomalies_timeseries_df.loc[393,'anomaly_label'],'spike_mv')
        
        
        for i in range(2160,2836):
            self.assertEqual(all_mv_anomalies_timeseries_df.loc[i,'anomaly_label'],'step_mv')
         
        for i in range(960,1637):
            self.assertEqual(all_mv_anomalies_timeseries_df.loc[i,'anomaly_label'],'pattern_mv')
           
        for i in range(576,768):
            self.assertEqual(all_mv_anomalies_timeseries_df.loc[i,'anomaly_label'],'noise_mv')
       
        for i in range(480,576):
        	self.assertEqual(all_mv_anomalies_timeseries_df.loc[i,'anomaly_label'],'clouds')

        
        
    def test_generate_time_boundaries(self):
       
        time_boundaries = generate_time_boundaries()
        self.assertIsInstance(time_boundaries,list)
        self.assertIsNotNone(time_boundaries[0])
        self.assertIsNotNone(time_boundaries[1])


 
        
        
        
        
        
    
                
        
	