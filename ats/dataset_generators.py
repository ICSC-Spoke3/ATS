from .timeseries_generators import HumiTempTimeseriesGenerator
import random as rnd
import pandas as pd
import itertools

# Setup logging
import logging
logger = logging.getLogger(__name__)

class DatasetGenerator():
    pass

class HumiTempDatasetGenerator(DatasetGenerator):

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

    def generate(self, n=9, time_span=None, plot=False,
                               effects='default', random_effects=[],
                               anomalies='default'):
        """
        Generate a synthetic dataset of humidity-temperature time series
        with different anomaly configurations.
        The dataset is divided into three groups, 0, 1, and 2 anomalies per series.
        Args:
            n (int, optional): Total number of series (must be multiple of 3).
            time_span (str, optional): Time window length (e.g. '30D', '60D').
            effects (list[str], optional): Effects that you can apply in each series (None, 'noise', 'seasons', 'clouds').
            random_effects (bool, optional): Random effects to apply across series.
            anomalies (list[str], optional): Anomalies to apply in each series.

        Returns:
            list: Generated synthetic time series.
        """
        if not isinstance(n, int):
            raise TypeError(f"`n` must be an integer, got {type(n).__name__}.")
        if n <= 0:
            raise ValueError("`n` must be a positive integer.")

        # Validate and convert parameters to lists
        effects = self.__check_list(effects, "effects")
        random_effects = self.__check_list(random_effects, "random_effects")
        anomalies = self.__check_list(anomalies, "anomalies")

        number_of_anomalies = len(anomalies)

        if number_of_anomalies == 0:
            logger.info("No anomalies specified; generating dataset without anomalies.")
        if number_of_anomalies == 1:
            logger.info("Single anomaly specified; generating dataset with 0 or 1 anomaly per series.")
        if number_of_anomalies >= 2:
            logger.info("Multiple anomalies specified; generating dataset with 0, 1, or 2 anomalies per series.")

        if number_of_anomalies == 2:
            anomaly1, anomaly2 = anomalies[0], anomalies[1]
            base1 = anomaly1.replace('_uv', '').replace('_mv', '')
            base2 = anomaly2.replace('_uv', '').replace('_mv', '')
            if (base1 == base2 and 
                ((anomaly1.endswith('_uv') and anomaly2.endswith('_mv')) or 
                 (anomaly1.endswith('_mv') and anomaly2.endswith('_uv')))):
                raise ValueError(f"Incompatible anomaly pair: {anomalies}. '{anomaly1}' and '{anomaly2}' cannot be used together.")
            
        if "clouds" in anomalies:
            if "clouds" not in effects:
                raise ValueError("Cannot use 'clouds' anomaly without including 'clouds' effect.") 

        dataset = []
        self._current_time_span = time_span or self.time_span
        
        try:
            generator = HumiTempTimeseriesGenerator(
                    temperature=self.temperature,
                    humidity=self.humidity,
                    sampling_interval=self.sampling_interval,
                    time_span=self._current_time_span
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing HumiTempTimeseriesGenerator") from e
        
   
        for i in range(n):
            if i % 2 == 1:
                anomalies_for_group = []
            else:
                if number_of_anomalies == 0:
                    anomalies_for_group = []
                elif number_of_anomalies == 1:
                    anomalies_for_group = rnd.sample(anomalies, 1)
                else:  # number_of_anomalies >= 2
                    if i % 4 == 0:
                        anomalies_for_group = rnd.sample(anomalies, 1)
                    else:
                        anomalies_for_group = rnd.sample(anomalies, 2)

            random_applied_effects = rnd.sample(random_effects, rnd.randint(0, len(random_effects))) 
            applied_effects = list(set(effects + random_applied_effects))

            try:
                series = generator.generate(effects=applied_effects or [],
                                            anomalies=anomalies_for_group or [], 
                                            plot=plot, generate_csv=False)
            except Exception as Error:
                logger.warning(f"Error generating dataset with anomalies {anomalies_for_group}: Retrying.")
                # Try other combinations of anomalies
                for combo in rnd.sample(list(itertools.combinations(anomalies, 2)), len(anomalies)):
                    try:
                        series = generator.generate(effects=applied_effects or [],
                                                    anomalies=list(combo), 
                                                    plot=True, generate_csv=False)
                        break  # Exit loop if successful
                    except Exception as e:
                        logger.warning(f"Failed with combination {combo}: {e}")
            logger.info(f"Generated dataset {len(dataset)+1} with effects: {applied_effects}")
            dataset.append(series)
    
        return dataset

    def _expected_points(self): 
        obs_window = pd.Timedelta(self._current_time_span)
        samp_interval = pd.Timedelta(self.sampling_interval)
        return int(obs_window / samp_interval)
