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
    

    def __generate_dataset__(self, howmany_series, 
                            observation_window, 
                            effects=[], 
                            randomize_effects=False, 
                            anomalies=[]):
        """
        Generic dataset generator used by all public dataset methods.
        """
        # It would be nice to have a function of synthetic data to achieve these:
        available_effects = ['noise', 'seasons', 'clouds']
        dataset = []

        self._current_observation_window = observation_window or self.observation_window
        
        if not isinstance(howmany_series, int) or howmany_series <= 0:
            raise ValueError(f"`howmany_series` must be a positive integer, got {howmany_series!r}.")
        
        if effects is None:
            effects = []
        if not isinstance(effects, list):
            raise TypeError(f"`effects` must be a list of strings, got {type(effects).__name__}.")

        try:
            generator = SyntheticHumiTempTimeseriesGenerator(
                    temperature=self.temperature,
                    humidity=self.humidity,
                    sampling_interval=self.sampling_interval,
                    observation_window=self._current_observation_window
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing SyntheticHumiTempTimeseriesGenerator") from e
        
        for i in range(howmany_series):
            chosen_effects = (
                rnd.sample(available_effects, rnd.randint(0, len(available_effects)))
                if randomize_effects else effects
            )
            try:
                series = generator.generate(effects=chosen_effects,
                                                    anomalies=anomalies or [], 
                                                    plot=False, 
                                                    generate_csv=False)
            except Exception as e:
                raise RuntimeError(f"Error generating synthetic series {i+1} with effects {chosen_effects}") from e
            
            logger.info(f"Generated dataset {i+1} with effects: {chosen_effects}")
            dataset.append(series)

        return dataset


    def generate_reference_dataset(self, howmany_series=3, observation_window=None, 
                                   effects=[],randomize_effects=False):
        """
        Generate a synthetic reference dataset composed of multiple humidity-temperature 
        time series, optionally with environmental effects applied.

        Args:
            howmany_series (int, opt): Number of series to generate (default = 3).
            observation_window (int, opt): Length of each time window.
            effects (list[str], opt): Effects to apply (['noise', 'seasons', 'clouds']).
            randomize_effects (bool, opt): Randomly choose effects for each series.

        Returns:
            list: Generated synthetic time series.
        """        
        reference_dataset = self.__generate_dataset__(
            howmany_series=howmany_series,
            observation_window=observation_window,
            effects=effects,
            randomize_effects=randomize_effects
        )
        return reference_dataset


    # Implemented for testing purposes                               
    def __expected_points__(self): 
        obs_window = pd.Timedelta(self._current_observation_window)
        samp_interval = pd.Timedelta(self.sampling_interval)
        return int(obs_window / samp_interval)

    def generate_reference_dataset(self, howmany_series=3, observation_window=None, 
                                   effects=[],randomize_effects=False):
        """
        Generate a synthetic reference dataset composed of multiple humidity-temperature 
        time series, optionally with environmental effects applied.

        Args:
            howmany_series (int, opt): Number of series to generate (default = 3).
            observation_window (int, opt): Length of each time window.
            effects (list[str], opt): Effects to apply (['noise', 'seasons', 'clouds']).
            randomize_effects (bool, opt): Randomly choose effects for each series.

        Returns:
            list: Generated synthetic time series.
        """        
        reference_dataset = self.__generate_dataset__(
            howmany_series=howmany_series,
            observation_window=observation_window,
            effects=effects,
            randomize_effects=randomize_effects
        )
        return reference_dataset


    # Implemented for testing purposes                               
    def __expected_points__(self): 
        obs_window = pd.Timedelta(self._current_observation_window)
        samp_interval = pd.Timedelta(self.sampling_interval)
        return int(obs_window / samp_interval)


    def generate_test_dataset(self, howmany_series=9, observation_window=None,
                               effects=[], randomize_effects=False):
        """
        Generate a synthetic test dataset of humidity-temperature time series
        with different anomaly configurations.

        The dataset is divided into three clusters:
          - Cluster 1: no anomalies
          - Cluster 2: univariate spike anomalies
          - Cluster 3: spike + step anomalies

        Args:
            howmany_series (int, optional): Total number of series (must be multiple of 3).
            observation_window (str, optional): Time window length (e.g. '30D', '60D').
            effects (list[str], optional): Environmental effects (['noise', 'seasons', 'clouds']).
            randomize_effects (bool, optional): Randomize effects across series.

        Returns:
            list: Generated synthetic time series.
        """
        if howmany_series <= 0 or howmany_series % 3 != 0:
            raise ValueError("`howmany_series` must be a positive multiple of 3 to form clusters.")

        num_cluster = 3
        howmany_series_for_cluster = howmany_series // num_cluster

        test_dataset = []

        anomalies_per_cluster = [
            [],  # 0 anomalies
            ['spike_uv'],  # 1 anomaly
            ['spike_uv', 'step_uv']  # 2 anomalies
        ]

        for i in range(num_cluster):
            series_cluster = self.__generate_dataset__(
                howmany_series=howmany_series_for_cluster,
                observation_window=observation_window,
                effects=effects,
                randomize_effects=randomize_effects,
                anomalies=anomalies_per_cluster[i]
                ) 
            test_dataset.extend(series_cluster)
        return test_dataset
