from .timeseries_generators import HumiTempTimeseriesGenerator, plot_func
import random as rnd
import pandas as pd

# Setup logging
import logging
logger = logging.getLogger(__name__)

class DatasetGenerator():
    pass

class HumiTempDatasetGenerator(DatasetGenerator):

    def __init__(self, temperature=True, humidity=True,
                 sampling_interval='15min'):
        self.temperature = temperature
        self.humidity = humidity
        self.sampling_interval = sampling_interval
    
    def __check_list(self, value, name):
        """
        Helper function to check and convert a value to a list.
        """
        if value is None:
            value = []
        if value=='default':
            raise ValueError(f'Default are not defined. You must explicitly provide the {value} to apply. Also None or empty list is accepted.')
        if not isinstance(value, list):
            raise TypeError(f"`{name}` must be a list, got {type(value).__name__}.")
        return value

    def _divide_time_interval(self,interval_str, max_anomalies_per_series, anomalies=[]):
        # TODO: Clarify UTC only
        total_seconds = int(pd.Timedelta(interval_str).total_seconds())
        segment_seconds = total_seconds / max_anomalies_per_series

        if segment_seconds < 20*24*60*60 and ("step_uv" in anomalies or "step_mv" in anomalies):
            raise NotImplementedError("Step anomalies require longer time_span")

        return "{}s".format(segment_seconds)
    
    def _generate_series(self,sampling_interval='15min',sub_time_span='30D', anomalies=[], effects=[],max_anomalies_per_series=2):
        if len(anomalies) == 0:
            n=0
        else:
            n=1
        series_combined = HumiTempTimeseriesGenerator(sampling_interval=sampling_interval,time_span=sub_time_span).generate(
            effects=effects,       
            anomalies=rnd.sample(anomalies, n) 
        )
        last_time = series_combined.index[-1] + pd.Timedelta(sampling_interval)

        for i in range(1, max_anomalies_per_series):
            last_time = series_combined.index[-1] + pd.Timedelta(sampling_interval)
            series = HumiTempTimeseriesGenerator( sampling_interval=sampling_interval,time_span=sub_time_span,
                                starting_year = last_time.year,
                                starting_month = last_time.month,
                                starting_day = last_time.day,
                                starting_hour = last_time.hour,
                                starting_minute = last_time.minute
            ).generate(
                effects=effects,       
                anomalies=rnd.sample(anomalies, n)
            )
            series_combined = pd.concat([series_combined, series])
        
        return series_combined   


    def generate(self, n_series=9, time_span='60D',
                 effects='default', anomalies='default', 
                 max_anomalies_per_series = 1, anomalies_ratio = 0.5):
        """
        Generate a synthetic dataset of humidity-temperature time series
        with different anomaly configurations.
        The dataset is divided alternates series with anomalies and series without it, based on `anomalies_ratio`.
        Args:
            n_series (int): Total number of series.
            time_span (str): Time window length (e.g. '30D', '60D').
            effects (list[str]): Effects that you can apply in each series (None, 'noise', 'seasons', 'clouds').
            anomalies (list[str]): Anomalies to apply in each series.
            max_anomalies_per_series (int): Max anomalies per series.
            anomalies_ratio (float): ratio of series with anomalies w.r.t. series without it in the dataset (0-1 range).
        Returns:
            list: Generated synthetic time series.
        """
        random_effects = [] # random_effects (bool, optional): Random effects to apply across series.
        n = n_series

        # Validate input parameters
        if not isinstance(n, int):
            raise TypeError(f"'n' must be an integer, got {type(n).__name__}.")
        if n <= 0:
            raise ValueError("'n' must be a positive integer.")
        if not isinstance(max_anomalies_per_series, int):
            raise TypeError(f"'max_anomalies_per_series' must be an integer, got {type(max_anomalies_per_series).__name__}.")
        if max_anomalies_per_series < 0:
            raise ValueError("'max_anomalies_per_series' must be a non-negative integer.")
        if not isinstance(anomalies_ratio, (int, float)):
            raise TypeError(f"'anomalies_ratio' must be a float, got {type(anomalies_ratio).__name__}.")
        if not (0 <= anomalies_ratio <= 1):
            raise ValueError("'anomalies_ratio' must be between 0 and 1.")
        
        # Validate list parameters
        effects = self.__check_list(effects, "effects")
        random_effects = self.__check_list(random_effects, "random_effects")
        anomalies = self.__check_list(anomalies, "anomalies")

        number_of_anomalies = len(anomalies)
        if number_of_anomalies == 0:
            logger.info("No anomalies specified; generating dataset without anomalies. \n " \
            "set max_anomalies_per_series to 0.")
            max_anomalies_per_series = 0
            sub_time_span = time_span
        if number_of_anomalies > 0:
            logger.info("Generating datest with max {} anomalies per series and " \
            "with a {} % of series without anomalies.".format(max_anomalies_per_series, anomalies_ratio * 100))
            max_anomalies_per_series = min(max_anomalies_per_series, number_of_anomalies)
            sub_time_span = self._divide_time_interval(time_span, max_anomalies_per_series,anomalies=anomalies)
        
        if "clouds" in anomalies:
            if "clouds" not in effects:
                raise ValueError("Cannot use 'clouds' anomaly without including 'clouds' effect.")

        dataset = []
        self.anomalies_list_per_series = []
        self.time_span = time_span
        self.dataset = dataset      

        accumulator=0.0
        for i in range(n):
            accumulator += anomalies_ratio
            if accumulator < 1.0:
                anomalies_for_group = []
            else:
                accumulator -= 1.0
                if number_of_anomalies == 0:
                    anomalies_for_group = []
                else:
                    number_of_anomalies = rnd.randint(1, max_anomalies_per_series)
                    anomalies_for_group = rnd.sample(anomalies, number_of_anomalies)

            random_applied_effects = rnd.sample(random_effects, rnd.randint(0, len(random_effects))) 
            applied_effects = list(set(effects + random_applied_effects))
            
            try:
                series = self._generate_series(sampling_interval=self.sampling_interval,
                                        sub_time_span=sub_time_span,
                                        anomalies=anomalies_for_group,
                                        effects=applied_effects,
                                        max_anomalies_per_series=max_anomalies_per_series)
            except Exception as e:
                logger.error(f"Error generating series {i+1}: {e}")
                continue
            logger.info(f"Generated dataset {len(dataset)+1} with effects: {applied_effects}")
            self.anomalies_list_per_series.append(anomalies_for_group)
            dataset.append(series)
    
        return dataset
    
    def plot_dataset(self):
        for df, anomalies in zip(self.dataset, self.anomalies_list_per_series):
            plot_func(df, anomalies=anomalies) 

    def _expected_points(self): 
        obs_window = pd.Timedelta(self.time_span)
        samp_interval = pd.Timedelta(self.sampling_interval)
        return int(obs_window / samp_interval)
