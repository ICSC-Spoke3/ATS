from .synthetic_data import SyntheticHumiTempTimeseriesGenerator
import random as rnd
import pandas as pd

# Setup logging
import logging
logger = logging.getLogger(__name__)

class EvaluationDataGenerator():
    pass

class HumiTempEvaluationDataGenerator(EvaluationDataGenerator):

    def __init__(self,# Series_lenght è ridodante
                 temperature=True, humidity=True,
                 sampling_interval= '15min',
                 observation_window='30D'):
        self.temperature = temperature
        self.humidity = humidity
        self.sampling_interval = sampling_interval
        self.observation_window = observation_window

    def generate_reference_datasets(self, howmany_datasets=3):
        # It wuold be nice to have a fuction of syntetic data to achieve theese:
        available_effects = ['noise', 'seasons', 'clouds']
        reference_datasets = []

        generator = SyntheticHumiTempTimeseriesGenerator(
                sampling_interval=self.sampling_interval,
                observation_window=self.observation_window,
                temperature=self.temperature,
                humidity=self.humidity
        )

        for i in range(howmany_datasets):
            n_effects = rnd.randint(0, len(available_effects))
            chosen_effects = rnd.sample(available_effects, n_effects)
            reference_dataset = generator.generate(effects=chosen_effects,
                                                anomalies=[], 
                                                plot=False, 
                                                generate_csv=False)
            logger.info(f"Dataset {i+1} generato con effetti: {chosen_effects}")
            reference_datasets.append(reference_dataset)

        return reference_datasets

        # Implemented for testing purposes                               
    def __expected_points__(self): 
        obs_window = pd.Timedelta(self.observation_window)
        samp_interval = pd.Timedelta(self.sampling_interval)
    return int(obs_window / samp_interval)

    #def generate_test_data(self, dataset_test_size, effect_type=[]):
       # generator = SyntheticHumiTempTimeseriesGenerator()
        #test_data = generator.generate(dataset_test_size, effect_type)

#        return test_data
