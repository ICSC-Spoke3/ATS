from .synthetic_data import SyntheticHumiTempTimeseriesGenerator
import random as rnd

class EvaluationDataGenerator():
    pass

class HumiTempEvaluationDataGenerator(EvaluationDataGenerator):

    def __init__(self, howmany_dataset,# Series_lenght è ridodante
                 temperature=True, humidity=True,
                 sampling_interval= '15min',
                 observation_window='30D'):
        self.howmany_dataset = howmany_dataset
        self.temperature = temperature
        self.humidity = humidity
        self.sampling_interval = sampling_interval
        self.observation_window = observation_window

    def generate_reference_data(self, howmany_dataset=3):
        # It wuold be nice to have a fuction of syntetic data to achieve theese:
        aviable_effects = ['noise', 'seasons', 'clouds']
        reference_dataset = []

        generator = SyntheticHumiTempTimeseriesGenerator(
                sampling_interval=self.sampling_interval,
                observation_window=self.observation_window,
                temperature=self.temperature,
                humidity=self.humidity
        )

        for i in range(howmany_dataset):
            n_effects = rnd.randint(0, len(available_effects))
            chosen_effects = rnd.sample(available_effects, n_effects)
            
            reference_data = generator.generate(effects=chosen_effects
                                                anomalies = None, 
                                                plot = False, 
                                                generate_csv = False)
            reference_dataset.append(reference_data)


        return reference_dataset

    #def generate_test_data(self, dataset_test_size, effect_type=[]):
       # generator = SyntheticHumiTempTimeseriesGenerator()
        #test_data = generator.generate(dataset_test_size, effect_type)

#        return test_data
