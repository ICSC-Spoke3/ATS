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


def _calculate_model_scores(single_model_evaluation={},granularity='data_point'):
    dataset_anomalies = set()
    for sample in single_model_evaluation.keys():
        sample_anomalies = set(single_model_evaluation[sample].keys())
        dataset_anomalies.update(sample_anomalies)

    anomaly_scores = {}
    for anomaly in dataset_anomalies:
        anomaly_scores[anomaly] = 0
    if 'false_positives' not in dataset_anomalies:
        anomaly_scores['false_positives'] = 0.0

    for anomaly in dataset_anomalies:
        for sample in single_model_evaluation.keys():
            if anomaly in single_model_evaluation[sample].keys():
                anomaly_scores[anomaly] += single_model_evaluation[sample][anomaly]

    if granularity == 'series':
        samples_n = len(single_model_evaluation)
        for key in anomaly_scores.keys():
            anomaly_scores[key] /= samples_n

    return anomaly_scores


class Evaluator():
    def __init__(self,test_data):
        self.test_data = test_data

    def _copy_dataset(self,dataset,models):
        dataset_copies = []
        for i in range(len(models)):
            dataset_copy = deepcopy(dataset)
            dataset_copies.append(dataset_copy)
        return dataset_copies

    def evaluate(self,models={},granularity='data_point'):
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

        dataset_copies = self._copy_dataset(formatted_dataset,models)
        models_scores = {}
        j = 0
        for model_name,model in models.items():
            single_model_evaluation = {}
            flagged_dataset = _get_model_output(dataset_copies[j],model)
            for i,sample_df in enumerate(flagged_dataset):
                if granularity == 'data_point':
                    single_model_evaluation[f'sample_{i+1}'] = _point_granularity_evaluation(sample_df,anomaly_labels_list[i])
                if granularity == 'variable':
                    single_model_evaluation[f'sample_{i+1}'] = _variable_granularity_evaluation(sample_df,anomaly_labels_list[i])
                if granularity == 'series':
                    single_model_evaluation[f'sample_{i+1}'] = _series_granularity_evaluation(sample_df,anomaly_labels_list[i])
                
            models_scores[model_name] = _calculate_model_scores(single_model_evaluation,granularity=granularity)
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

def _variable_granularity_evaluation(flagged_timeseries_df,anomaly_labels_df):
    one_series_evaluation_result = {}
    flag_columns_n = len(flagged_timeseries_df.filter(like='anomaly').columns)
    variables_n = len(flagged_timeseries_df.columns) - flag_columns_n
    if variables_n != 1 and variables_n != flag_columns_n:
        raise ValueError('Variable granularity is not for this model')
    normalization_factor = variables_n * len(flagged_timeseries_df)

    for anomaly,frequency in anomaly_labels_df.value_counts(dropna=False).items():
        anomaly_count = 0
        for timestamp in flagged_timeseries_df.index:
            if anomaly_labels_df[timestamp] == anomaly:
                for column in flagged_timeseries_df.filter(like='anomaly').columns:
                    anomaly_count += flagged_timeseries_df.loc[timestamp,column]
        one_series_evaluation_result[anomaly] = anomaly_count / normalization_factor

    one_series_evaluation_result['false_positives'] = one_series_evaluation_result.pop(None)
    return one_series_evaluation_result

def _point_granularity_evaluation(flagged_timeseries_df,anomaly_labels_df):
    one_series_evaluation_result = {}
    normalization_factor = len(flagged_timeseries_df)

    for anomaly,frequency in anomaly_labels_df.value_counts(dropna=False).items():
        anomaly_count = 0
        for timestamp in flagged_timeseries_df.index:
            if anomaly_labels_df[timestamp] == anomaly:
                for column in flagged_timeseries_df.filter(like='anomaly').columns:
                    if flagged_timeseries_df.loc[timestamp,column]:
                        anomaly_count += 1
                        break
        one_series_evaluation_result[anomaly] = anomaly_count / normalization_factor

    one_series_evaluation_result['false_positives'] = one_series_evaluation_result.pop(None)
    return one_series_evaluation_result

def _series_granularity_evaluation(flagged_timeseries_df,anomaly_labels_df):
    anomalies = []
    for anomaly,frequency in anomaly_labels_df.value_counts(dropna=False).items():
        if anomaly is not None:
            anomalies.append(anomaly)
    anomalies_n = len(anomalies)
    if anomalies_n > 1:
        raise ValueError('Evaluation with series granularity supports series with only one anomaly')

    one_series_evaluation_result = {}
    is_series_anomalous = 0
    for timestamp in flagged_timeseries_df.index:
        for column in flagged_timeseries_df.filter(like='anomaly').columns:
            if flagged_timeseries_df.loc[timestamp,column]:
                is_series_anomalous = 1
                break
    if is_series_anomalous and not anomalies:
        one_series_evaluation_result['false_positives'] = 1
    elif is_series_anomalous and anomalies:
        one_series_evaluation_result[anomalies[0]] = 1
    elif not is_series_anomalous and anomalies:
        one_series_evaluation_result[anomalies[0]] = 0

    return one_series_evaluation_result