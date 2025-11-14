from .anomaly_detectors.naive import MinMaxAnomalyDetector
import pandas as pd
from copy import deepcopy

def _format_for_anomaly_detector(input_df,synthetic=False):
    if synthetic:
        input_df.drop(columns=['effect_label'],inplace=True)
    if 'anomaly_label' not in input_df.columns:
        raise ValueError('The input DataFrame has to contain an "anomaly_label" column for evaluation')

    anomaly_labels = input_df.loc[:,'anomaly_label']
    input_df.drop(columns=['anomaly_label'],inplace=True)
    return input_df,anomaly_labels

def evaluate_anomaly_detector(evaluated_timeseries_df, anomaly_labels, details=False):

    evaluated_anomaly_flags = evaluated_timeseries_df.filter(like='anomaly')
    if len(evaluated_anomaly_flags.columns) == 1 and len(evaluated_timeseries_df.columns)>2:
        raise NotImplementedError('The detector needs to flag anomalies for each quantity of the timeseries')
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


def _calculate_model_scores(single_model_evaluation={}):
    anomalies = list(single_model_evaluation['sample_1'].keys())
    samples_n = len(single_model_evaluation)
    detections_per_anomaly = {}
    avg_detections_per_anomaly = {}

    for anomaly in anomalies:
        detections_per_anomaly[anomaly] = 0

    for sample in single_model_evaluation.keys():
        for anomaly in single_model_evaluation[sample].keys():
            # TODO: evaluate_anomaly_detector and calculate_model_scores are redundant
            if single_model_evaluation[sample][anomaly] and anomaly != 'false_positives':
                detections_per_anomaly[anomaly] +=1
            elif anomaly == 'false_positives':
                detections_per_anomaly[anomaly] +=single_model_evaluation[sample][anomaly]

    for anomaly,counts in detections_per_anomaly.items():
        avg_detections_per_anomaly[anomaly] = counts/samples_n if anomaly != 'false_positives' else counts

    return avg_detections_per_anomaly


class Evaluator():
    def __init__(self,test_data):
        self.test_data = test_data

    def _copy_dataset(self,dataset):
        dataset_copies = []
        for i in range(len(self.models)):
            dataset_copy = deepcopy(dataset)
            dataset_copies.append(dataset_copy)
        return dataset_copies

    def evaluate(self,models={}):
        if not models:
            raise ValueError('There are no models to evaluate')
        if not self.test_data:
            raise ValueError('No input data set')

        formatted_dataset = []
        anomaly_labels_list = []
        for series in self.test_data:
            synthetic = 'effect_label' in series.columns
            formatted_series,anomaly_labels = _format_for_anomaly_detector(series,synthetic=synthetic)
            formatted_dataset.append(formatted_series)
            anomaly_labels_list.append(anomaly_labels)

        dataset_copies = self.copy_dataset(formatted_dataset)
        models_scores = {}
        j = 0
        for model_name,model in models.items():
            single_model_evaluation = {}
            flagged_dataset = get_model_output(dataset_copies[j],model)
            for i,sample_df in enumerate(flagged_dataset):
                single_model_evaluation[f'sample_{i+1}'] = evaluate_anomaly_detector(sample_df,anomaly_labels_list[i])
            models_scores[model_name] = calculate_model_scores(single_model_evaluation)
            j+=1

        return models_scores

def _get_model_output(dataset,model):
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
