from .synthetic_data import SyntheticHumiTempTimeseriesGenerator

class EvaluationDataGenerator():
    pass

class HumiTempEvaluationDataGenerator(EvaluationDataGenerator):

    def __init__(self, howmany_series,# Series_lenght è ridodante
                 temperature=True, humidity=True,
                 sampling_interval= '15min',
                 observation_window='30D'):
        self.howmany_series = howmany_series
        self.temperature = temperature
        self.humidity = humidity
        self.sampling_interval = sampling_interval
        self.observation_window = observation_window

    def generate_reference_data(self, dataset_reference_size=3, effect_type=[]):
        generator = SyntheticHumiTempTimeseriesGenerator()
        reference_data = generator.generate(dataset_reference_size, 
                                            effect_type)

        return reference_data

    #def generate_test_data(self, dataset_test_size, effect_type=[]):
       # generator = SyntheticHumiTempTimeseriesGenerator()
        #test_data = generator.generate(dataset_test_size, effect_type)

#        return test_data
