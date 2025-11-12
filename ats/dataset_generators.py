from .synthetic_data import SyntheticHumiTempTimeseriesGenerator
import random as rnd
import pandas as pd

# Setup logging
import logging
logger = logging.getLogger(__name__)

class SynteticDatasetGenerator():
    pass

class SynteticHumiTempDatasetGenerator(SynteticDatasetGenerator):

    def __init__(self, temperature=True, humidity=True,
                 sampling_interval='15min', time_span='30D'):
        self.temperature = temperature
        self.humidity = humidity
        self.sampling_interval = sampling_interval
        self.time_span = time_span
        self._current_time_span = time_span
    
    def __check_list(self, value, name):
        """
        Helper function to check and convert a value to a list.
        """
        if value is None:
            value = []
        if value=='default':
            raise NotImplementedError(f'You must explicitly provide the {value} to apply. Also None or Empty is accepted.')
        if not isinstance(value, list):
            raise TypeError(f"`{name}` must be a list, got {type(value).__name__}.")
        return value

    def _generate_dataset(self, n,
                            time_span,
                            effects=[],
                            random_effects=[],
                            anomalies=[],
                            random_anomalies=[]
                            ):
        """
        Generic dataset generator used by all public dataset methods.
        """
        dataset = []

        # The following line serves for calculating expected points in tests
        self._current_time_span = time_span or self.time_span
        
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"`n` must be a positive integer, got {n!r}.")

        effects = self.__check_list(effects, "effects")
        random_effects = self.__check_list(random_effects, "random_effects")
        anomalies = self.__check_list(anomalies, "anomalies")
        random_anomalies = self.__check_list(random_anomalies, "random_anomalies")

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
            random_applied_effects = rnd.sample(random_effects, rnd.randint(0, len(random_effects))) 
            # Eliminate duplicates and combine effects
            applied_effects = list(set(effects + random_applied_effects))
            try:
                series = generator.generate(effects=applied_effects or [],
                                            anomalies=anomalies or [], 
                                            plot=False, generate_csv=False)
            except Exception as e:
                raise RuntimeError(f"Error generating synthetic series {i+1} with effects {applied_effects}") from e

            logger.info(f"Generated dataset {i+1} with effects: {applied_effects}")
            dataset.append(series)

        return dataset


    def generate_reference_dataset(self, n=3, time_span=None, 
                                   effects='default', random_effects=[]):
        """
        Generate a synthetic reference dataset composed of multiple humidity-temperature 
        time series, optionally with environmental effects applied.

        Args:
            n (int, opt): Number of series to generate (default = 3).
            time_span (int, opt): Length of each time window.
            effects (list[str], opt): Effects that you can apply in each series (None, 'noise', 'seasons', 'clouds').
            random_effects (list[str], opt): Random effects to apply across series.

        Returns:
            list: Generated synthetic time series.
        """        
        reference_dataset = self._generate_dataset(
            n=n,
            time_span=time_span,
            effects=effects,
            random_effects=random_effects
        )
        return reference_dataset

    # Implemented for testing purposes                               
    def _expected_points(self): 
        obs_window = pd.Timedelta(self._current_time_span)
        samp_interval = pd.Timedelta(self.sampling_interval)
        return int(obs_window / samp_interval)

    def generate_test_dataset(self, n=9, time_span=None,
                               effects='default', random_effects=[],
                               anomalies='default', random_anomalies=None):
        """
        Generate a synthetic test dataset of humidity-temperature time series
        with different anomaly configurations.
        The dataset is divided into three groups, 0, 1, and 2 anomalies per series.
        Args:
            n (int, optional): Total number of series (must be multiple of 3).
            time_span (str, optional): Time window length (e.g. '30D', '60D').
            effects (list[str], optional): Effects that you can apply in each series (None, 'noise', 'seasons', 'clouds').
            random_effects (bool, optional): Random effects to apply across series.
            anomalies (list[str], optional): Anomalies to apply in each series.
            random_anomalies (list[str], optional): Random anomalies to apply across series.

        Returns:
            list: Generated synthetic time series.
        """
        if n <= 0 or n % 3 != 0:
            raise ValueError("`n` must be a positive multiple of 3 to form groups.")

        test_dataset = []

        if len(anomalies)<2:
            raise ValueError("Define at least two anomaliesfor generating test datasets with 1 and 2 anomalies per series.")

        if random_anomalies is not None and random_anomalies != []:
            raise NotImplementedError("Random anomalies not yet implemented.")
        
        num_group = 3
        n_per_group = n // num_group
        for i in range(num_group):
            if i % 3 == 0:
                anomalies_for_group = []
            elif i % 3 == 1:
                anomalies_for_group = rnd.sample(anomalies, 1)
            else:
                anomalies_for_group = rnd.sample(anomalies, 2)
            series_group = self._generate_dataset(
                n=n_per_group,
                time_span=time_span,
                effects=effects,
                random_effects=random_effects,
                anomalies=anomalies_for_group,
                random_anomalies=random_anomalies,
                ) 
            test_dataset.extend(series_group)
        return test_dataset
