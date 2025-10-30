from .synthetic_data import SyntheticHumiTempTimeseriesGenerator
import random as rnd
import pandas as pd

# Setup logging
import logging
logger = logging.getLogger(__name__)

class EvaluationDataGenerator():
    pass

class HumiTempEvaluationDataGenerator(EvaluationDataGenerator):

    def __init__(self, temperature=True, humidity=True,
                 sampling_interval='15min', observation_window='30D'):
        self.temperature = temperature
        self.humidity = humidity
        self.sampling_interval = sampling_interval
        self.observation_window = observation_window
        self._current_observation_window = observation_window

    def generate_reference_dataset(self, howmany_series=3, observation_window=None, 
                                   effects=[],randomize_effects=False):
        # It would be nice to have a function of synthetic data to achieve these:
        available_effects = ['noise', 'seasons', 'clouds']
        reference_dataset = []

        self._current_observation_window = observation_window or self.observation_window

        generator = SyntheticHumiTempTimeseriesGenerator(
                temperature=self.temperature,
                humidity=self.humidity,
                sampling_interval=self.sampling_interval,
                observation_window=self._current_observation_window
        )

        for i in range(howmany_series):
            if randomize_effects:
                n_effects = rnd.randint(0, len(available_effects))
                chosen_effects = rnd.sample(available_effects, n_effects)
            else:
                chosen_effects = effects
            reference_series = generator.generate(effects=chosen_effects,
                                                anomalies=[], 
                                                plot=False, 
                                                generate_csv=False)
            logger.info(f"Generated dataset {i+1} with effects: {chosen_effects}")
            reference_dataset.append(reference_series)

        return reference_dataset

    # Implemented for testing purposes                               
    def __expected_points__(self): 
        obs_window = pd.Timedelta(self._current_observation_window)
        samp_interval = pd.Timedelta(self.sampling_interval)
        return int(obs_window / samp_interval)

    #def generate_test_data(self, dataset_test_size, effect_type=[]):
       # generator = SyntheticHumiTempTimeseriesGenerator()
        #test_data = generator.generate(dataset_test_size, effect_type)

#        return test_data
