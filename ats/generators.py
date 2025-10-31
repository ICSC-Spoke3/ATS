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
                 sampling_interval='15min', time_span='30D'):
        self.temperature = temperature
        self.humidity = humidity
        self.sampling_interval = sampling_interval
        self.time_span = time_span
        self._current_time_span = time_span
    

    def _generate_dataset(self, n, 
                            time_span, 
                            effects=[], 
                            randomize_effects=False, 
                            anomalies=[]):
        """
        Generic dataset generator used by all public dataset methods.
        """
        # It would be nice to have a function of synthetic data to achieve these:
        available_effects = effects
        dataset = []

        self._current_time_span = time_span or self.time_span
        
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"`n` must be a positive integer, got {n!r}.")
        
        if effects is None:
            effects = []
        if not isinstance(effects, list):
            raise TypeError(f"`effects` must be a list of strings, got {type(effects).__name__}.")

        try:
            generator = SyntheticHumiTempTimeseriesGenerator(
                    temperature=self.temperature,
                    humidity=self.humidity,
                    sampling_interval=self.sampling_interval,
                    time_span=self._current_time_span
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing SyntheticHumiTempTimeseriesGenerator") from e
        
        for i in range(n):
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


    def generate_reference_dataset(self, n=3, time_span=None, 
                                   effects='default',randomize_effects=False):
        """
        Generate a synthetic reference dataset composed of multiple humidity-temperature 
        time series, optionally with environmental effects applied.

        Args:
            n (int, opt): Number of series to generate (default = 3).
            time_span (int, opt): Length of each time window.
            effects (list[str], opt): Effects to apply (['noise', 'seasons', 'clouds']).
            randomize_effects (bool, opt): Randomly choose effects for each series.

        Returns:
            list: Generated synthetic time series.
        """
        if effects=='default':
            raise NotImplementedError('You must explicitilty provides the effects____AAAAA')
        
        reference_dataset = self._generate_dataset(
            n=n,
            time_span=time_span,
            effects=effects,
            randomize_effects=randomize_effects
        )
        return reference_dataset

    # Implemented for testing purposes                               
    def _expected_points(self): 
        obs_window = pd.Timedelta(self._current_time_span)
        samp_interval = pd.Timedelta(self.sampling_interval)
        return int(obs_window / samp_interval)

    def generate_test_dataset(self, n=9, time_span=None,
                               effects=[], randomize_effects=False):
        """
        Generate a synthetic test dataset of humidity-temperature time series
        with different anomaly configurations.

        The dataset is divided into three groups:
          - group 1: no anomalies
          - group 2: univariate spike anomalies
          - group 3: spike + step anomalies

        Args:
            n (int, optional): Total number of series (must be multiple of 3).
            time_span (str, optional): Time window length (e.g. '30D', '60D').
            effects (list[str], optional): Environmental effects (['noise', 'seasons', 'clouds']).
            randomize_effects (bool, optional): Randomize effects across series.

        Returns:
            list: Generated synthetic time series.
        """
        if n <= 0 or n % 3 != 0:
            raise ValueError("`n` must be a positive multiple of 3 to form groups.")

        num_group = 3
        n_for_group = n // num_group

        test_dataset = []

        anomalies_per_group = [
            [],  # 0 anomalies
            ['spike_uv'],  # 1 anomaly
            ['spike_uv', 'step_uv']  # 2 anomalies
        ]

        for i in range(num_group):
            series_group = self._generate_dataset(
                n=n_for_group,
                time_span=time_span,
                effects=effects,
                randomize_effects=randomize_effects,
                anomalies=anomalies_per_group[i]
                ) 
            test_dataset.extend(series_group)
        return test_dataset
