from .anomaly_detectors.naive import MinMaxAnomalyDetector
import pandas as pd
from copy import deepcopy

def _format_for_anomaly_detector(input_df,synthetic=False):
    if synthetic:
        input_df.drop(columns=['effect_label'],inplace=True)

    anomaly_labels = input_df.loc[:,'anomaly_label']
    input_df.drop(columns=['anomaly_label'],inplace=True)
    return input_df,anomaly_labels

def evaluate_anomaly_detector(anomaly_detector, evaluation_timeseries_df, synthetic=False,details=False):

    if not isinstance(anomaly_detector,MinMaxAnomalyDetector):
        raise ValueError('Only MinMaxAnomalyDetector is supported')
    if 'anomaly_label' not in evaluation_timeseries_df.columns:
        raise ValueError('The anomaly_label column is missing: it is necessary for evaluation')

    evaluation_timeseries_df, anomaly_labels = _format_for_anomaly_detector(evaluation_timeseries_df,synthetic=synthetic)
    evaluated_timeseries_df = anomaly_detector.apply(evaluation_timeseries_df)
    evaluated_anomaly_flags = evaluated_timeseries_df.filter(like='_anomaly')
    evaluation_results = {}
    evaluation_details = {}
    for anomaly_label,frequency in anomaly_labels.value_counts(dropna=False).items():
        anomaly_label_counts = 0
        for time_index in evaluated_timeseries_df.index:
            if anomaly_labels.loc[time_index] == anomaly_label:
                for column in evaluated_anomaly_flags.columns:
                    is_anomalous_value = evaluated_anomaly_flags.loc[time_index,column]
                    if is_anomalous_value:
                        if not anomaly_label_counts:
                            evaluation_details[anomaly_label]={ time_index: {quantity: bool(evaluated_anomaly_flags.loc[time_index,quantity]) for quantity in evaluated_anomaly_flags.columns}}
                        else:
                            evaluation_details[anomaly_label][time_index]={quantity: bool(evaluated_anomaly_flags.loc[time_index,quantity]) for quantity in evaluated_anomaly_flags.columns}
                        anomaly_label_counts += 1
                        evaluation_results[anomaly_label] = True if anomaly_label is not None else anomaly_label_counts
        if not anomaly_label_counts:
            evaluation_results[anomaly_label] = False if anomaly_label is not None else 0

    if None in evaluation_results.keys():
        evaluation_results['false_positives'] = evaluation_results.pop(None)
    if None in evaluation_details.keys():
        evaluation_details['false_positives'] = evaluation_details.pop(None)

    if details:
        return evaluation_results,evaluation_details
    else:
        return evaluation_results

class Evaluator():
    def __init__(self,test_data,models):
        self.test_data = test_data
        self.models = models

    @staticmethod
    def calculate_model_scores(single_model_evaluation={}):
        anomalies = list(single_model_evaluation['sample_1'].keys())
        samples_n = len(single_model_evaluation)
        detections_per_anomaly = {}
        for anomaly in anomalies:
            detections_per_anomaly[anomaly] = 0

        for sample in single_model_evaluation.keys():
            for anomaly in single_model_evaluation[sample].keys():
                if single_model_evaluation[sample][anomaly] and anomaly != 'false_positives':
                    detections_per_anomaly[anomaly] +=1
                elif anomaly == 'false_positives':
                    detections_per_anomaly[anomaly] +=single_model_evaluation[sample][anomaly]

        avg_detections_per_anomaly = {anomaly: counts/samples_n for anomaly, counts in detections_per_anomaly.items()}
        return avg_detections_per_anomaly

    def copy_dataset(self):
        dataset_copies = []
        for i in range(len(self.models)):
            dataset_copy = deepcopy(self.test_data)
            dataset_copies.append(dataset_copy)
        return dataset_copies

    def evaluate(self):
        if not self.models:
            raise ValueError('There are no models to evaluate')
        if not self.test_data:
            raise ValueError('No input data set')

        dataset_copies = self.copy_dataset()
        models_scores = {}
        j = 0
        for model_name,model in self.models.items():
            single_model_evaluation = {}
            for i,sample_df in enumerate(dataset_copies[j]):
                single_model_evaluation[f'sample_{i+1}'] = evaluate_anomaly_detector(model,dataset_copies[j][i],synthetic=synthetic)
            models_scores[model_name] = calculate_model_scores(single_model_evaluation)
            j+=1

        return pd.Dataframe(models_scores)

def get_model_output(dataset,model):
    if not isinstance(dataset,list):
        raise ValueError('The input dataset has to be a list')
    for series in dataset:
        if not isinstance(series,pd.DataFrame):
            raise ValueError('Dataset elements have to be a pandas DataFrame')

    flagged_dataset = []
    try:
        flagged_dataset = model.apply(dataset)
    except NotImplementedError:
        for series in dataset:
            flagged_series = model.apply(series)
            flagged_dataset.append(flagged_series)

    return flagged_dataset
