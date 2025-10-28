from .synthetic_data import SyntheticHumiTempTimeseriesGenerator

class EvaluationDataGenerator():
    pass

class HumiTempEvaluationDataGenerator(EvaluationDataGenerator):
    
    def generate_reference_data(data_size,effect_type = []):
        generator = SyntheticHumiTempTimeseriesGenerator()
        reference_data = generator.generate(data_size, effect_type)
        return reference_data
