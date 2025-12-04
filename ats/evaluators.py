from .anomaly_detectors.naive import MinMaxAnomalyDetector
import pandas as pd
from copy import deepcopy

def _format_for_anomaly_detector(df,synthetic=False):
    if synthetic:
        df = df.drop(columns=['effect_label'],inplace=False)
    if 'anomaly_label' not in df.columns:
        raise ValueError('The input DataFrame has to contain an "anomaly_label" column for evaluation')

    anomaly_labels = df.loc[:,'anomaly_label']
    df = df.drop(columns=['anomaly_label'],inplace=False)
    return df,anomaly_labels

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
    dataset_anomalies = set()
    for sample in single_model_evaluation.keys():
        sample_anomalies = set(single_model_evaluation[sample].keys())
        dataset_anomalies.update(sample_anomalies)

    model_scores = {}
    anomalies_count = 0
    false_positives_count = 0
    anomalies_ratio = 0
    false_positives_ratio = 0
    anomalous_series_n = 0
    for sample in single_model_evaluation.keys():
        anomalies_count += single_model_evaluation[sample]['anomalies_count']
        if single_model_evaluation[sample]['anomalies_ratio'] is not None:
            anomalies_ratio += single_model_evaluation[sample]['anomalies_ratio']
            anomalous_series_n += 1
        false_positives_count += single_model_evaluation[sample]['false_positives_count']
        false_positives_ratio += single_model_evaluation[sample]['false_positives_ratio']

    model_scores['anomalies_count'] = anomalies_count
    if anomalous_series_n:
        model_scores['anomalies_ratio'] = anomalies_ratio/anomalous_series_n
    else:
        model_scores['anomalies_ratio'] = None
    model_scores['false_positives_count'] = false_positives_count
    model_scores['false_positives_ratio'] = false_positives_ratio/len(single_model_evaluation)

    return model_scores


class Evaluator():
    def __init__(self,test_data):
        self.test_data = test_data

    def _copy_dataset(self,dataset,models):
        dataset_copies = []
        for i in range(len(models)):
            dataset_copy = deepcopy(dataset)
            dataset_copies.append(dataset_copy)
        return dataset_copies

    def evaluate(self,models={},granularity='point',strategy='flags'):
        if strategy != 'flags':
            raise NotImplementedError(f'Evaluation strategy {strategy} is not implemented')

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
                if granularity == 'point':
                    single_model_evaluation[f'sample_{i+1}'] = _point_granularity_evaluation(sample_df,anomaly_labels_list[i])
                elif granularity == 'variable':
                    single_model_evaluation[f'sample_{i+1}'] = _variable_granularity_evaluation(sample_df,anomaly_labels_list[i])
                elif granularity == 'series':
                    single_model_evaluation[f'sample_{i+1}'] = _series_granularity_evaluation(sample_df,anomaly_labels_list[i])
                else:
                    raise ValueError(f'Unknown granularity {granularity}')
                
            models_scores[model_name] = _calculate_model_scores(single_model_evaluation)
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

    total_inserted_anomalies_n = 0
    total_detected_anomalies_n = 0
    detection_counts_by_anomaly_type = {}
    for anomaly,frequency in anomaly_labels_df.value_counts(dropna=False).items():
        if anomaly is not None:
            total_inserted_anomalies_n += frequency
        anomaly_count = 0
        for timestamp in flagged_timeseries_df.index:
            if anomaly_labels_df[timestamp] == anomaly:
                for column in flagged_timeseries_df.filter(like='anomaly').columns:
                    anomaly_count += flagged_timeseries_df.loc[timestamp,column]
        if anomaly is not None:
            total_detected_anomalies_n += anomaly_count
        detection_counts_by_anomaly_type[anomaly] = anomaly_count

    total_inserted_anomalies_n *= variables_n
    one_series_evaluation_result['false_positives_count'] = detection_counts_by_anomaly_type.pop(None)
    one_series_evaluation_result['false_positives_ratio'] = one_series_evaluation_result['false_positives_count']/normalization_factor
    one_series_evaluation_result['anomalies_count'] = total_detected_anomalies_n
    if total_inserted_anomalies_n:
        one_series_evaluation_result['anomalies_ratio'] = total_detected_anomalies_n/total_inserted_anomalies_n
    else:
        one_series_evaluation_result['anomalies_ratio'] = None
    return one_series_evaluation_result

def _point_granularity_evaluation(flagged_timeseries_df,anomaly_labels_df):
    one_series_evaluation_result = {}
    normalization_factor = len(flagged_timeseries_df)

    total_inserted_anomalies_n = 0
    total_detected_anomalies_n = 0
    detection_counts_by_anomaly_type = {}
    for anomaly,frequency in anomaly_labels_df.value_counts(dropna=False).items():
        if anomaly is not None:
            total_inserted_anomalies_n += frequency
        anomaly_count = 0
        for timestamp in flagged_timeseries_df.index:
            if anomaly_labels_df[timestamp] == anomaly:
                for column in flagged_timeseries_df.filter(like='anomaly').columns:
                    if flagged_timeseries_df.loc[timestamp,column]:
                        anomaly_count += 1
                        break
        if anomaly is not None:
            total_detected_anomalies_n += anomaly_count
        detection_counts_by_anomaly_type[anomaly] = anomaly_count
        one_series_evaluation_result[anomaly] = anomaly_count / normalization_factor

    one_series_evaluation_result['false_positives_count'] = detection_counts_by_anomaly_type.pop(None)
    one_series_evaluation_result['false_positives_ratio'] = one_series_evaluation_result['false_positives_count']/normalization_factor
    one_series_evaluation_result['anomalies_count'] = total_detected_anomalies_n
    if total_inserted_anomalies_n:
        one_series_evaluation_result['anomalies_ratio'] = total_detected_anomalies_n/total_inserted_anomalies_n
    else:
        one_series_evaluation_result['anomalies_ratio'] = None
    return one_series_evaluation_result

def _series_granularity_evaluation(flagged_timeseries_df,anomaly_labels_df):
    anomalies = []
    for anomaly,frequency in anomaly_labels_df.value_counts(dropna=False).items():
        if anomaly is not None:
            anomalies.append(anomaly)

    one_series_evaluation_result = {}
    is_series_anomalous = 0
    for timestamp in flagged_timeseries_df.index:
        for column in flagged_timeseries_df.filter(like='anomaly').columns:
            if flagged_timeseries_df.loc[timestamp,column]:
                is_series_anomalous = 1
                break
    one_series_evaluation_result['false_positives_count'] = 1 if is_series_anomalous and not anomalies else 0
    one_series_evaluation_result['false_positives_ratio'] = one_series_evaluation_result['false_positives_count']
    one_series_evaluation_result['anomalies_count'] = 1 if is_series_anomalous and anomalies else 0
    one_series_evaluation_result['anomalies_ratio'] = one_series_evaluation_result['anomalies_count'] if anomalies else None

    return one_series_evaluation_result