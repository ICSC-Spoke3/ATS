from .anomaly_detectors.naive import MinMaxAnomalyDetector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    model_scores = {}
    anomalies_count = 0
    false_positives_count = 0
    anomalies_ratio = 0
    false_positives_ratio = 0
    anomalous_series_n = 0
    for sample in single_model_evaluation.keys():
        anomalies_count += single_model_evaluation[sample]['true_positives_count']
        if single_model_evaluation[sample]['true_positives_rate'] is not None:
            anomalies_ratio += single_model_evaluation[sample]['true_positives_rate']
            anomalous_series_n += 1
        false_positives_count += single_model_evaluation[sample]['false_positives_count']
        false_positives_ratio += single_model_evaluation[sample]['false_positives_ratio']

    model_scores['true_positives_count'] = anomalies_count
    if anomalous_series_n:
        model_scores['true_positives_rate'] = anomalies_ratio/anomalous_series_n
    else:
        model_scores['true_positives_rate'] = None
    model_scores['false_positives_count'] = false_positives_count
    model_scores['false_positives_ratio'] = false_positives_ratio/len(single_model_evaluation)
    return model_scores

def _get_breakdown_info(single_model_evaluation={}):
    for sample in single_model_evaluation.keys():
        if 'true_positives_count' in single_model_evaluation[sample].keys():
            del single_model_evaluation[sample]['true_positives_count']
        if 'true_positives_rate' in single_model_evaluation[sample].keys():
            del single_model_evaluation[sample]['true_positives_rate']
        if 'false_positives_count' in single_model_evaluation[sample].keys():
            del single_model_evaluation[sample]['false_positives_count']
        if 'false_positives_ratio' in single_model_evaluation[sample].keys():
            del single_model_evaluation[sample]['false_positives_ratio']

    breakdown_info = {}
    # how many series in the dataset have that anomaly type
    anomaly_series_count_by_type = {}
    for sample, sample_evaluation in single_model_evaluation.items():
        for key in sample_evaluation.keys():
            if key in breakdown_info.keys():
                anomaly_series_count_by_type[key] +=1
                breakdown_info[key] += sample_evaluation[key]
            else:
                anomaly_series_count_by_type[key] =1
                breakdown_info[key] = sample_evaluation[key]

    for key in breakdown_info.keys():
        if '_rate' in key:
            breakdown_info[key] /= anomaly_series_count_by_type[key]

    return breakdown_info


class Evaluator():
    def __init__(self,test_data):
        self.test_data = test_data

    def _copy_dataset(self,dataset,models):
        dataset_copies = []
        for i in range(len(models)):
            dataset_copy = deepcopy(dataset)
            dataset_copies.append(dataset_copy)
        return dataset_copies

    def evaluate(self,models={},granularity='point',strategy='flags',breakdown=False):
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
                    if strategy == 'flags':
                        single_model_evaluation[f'sample_{i+1}'] = _point_granularity_evaluation(sample_df,anomaly_labels_list[i],breakdown=breakdown)
                    elif strategy == 'events':
                        single_model_evaluation[f'sample_{i+1}'] = _point_eval_with_events_strategy(sample_df,anomaly_labels_list[i],breakdown=breakdown)

                elif granularity == 'variable':
                    if strategy == 'flags':
                        single_model_evaluation[f'sample_{i+1}'] = _variable_granularity_evaluation(sample_df,anomaly_labels_list[i], breakdown = breakdown)
                    elif strategy == 'events':
                        single_model_evaluation[f'sample_{i+1}'] = _variable_eval_with_events_strategy(sample_df,anomaly_labels_list[i],breakdown=breakdown)

                elif granularity == 'series':
                    if strategy == 'flags':
                        single_model_evaluation[f'sample_{i+1}'] = _series_granularity_evaluation(sample_df,anomaly_labels_list[i], breakdown = breakdown)
                    elif strategy == 'events':
                        single_model_evaluation[f'sample_{i+1}'] = _series_eval_with_events_strategy(sample_df,anomaly_labels_list[i],breakdown=breakdown)

                else:
                    raise ValueError(f'Unknown granularity {granularity}')

            if breakdown:
                scores = _calculate_model_scores(single_model_evaluation)
                breakdown_info = _get_breakdown_info(single_model_evaluation)
                models_scores[model_name] = scores | breakdown_info
            else:
                models_scores[model_name] = _calculate_model_scores(single_model_evaluation)
            j+=1

        return models_scores

    @staticmethod
    def summary(evaluation, full=False):
        if full:
            df = pd.DataFrame.from_dict(evaluation, orient='index')
        else:
            df = pd.DataFrame.from_dict(evaluation, orient='index').filter(['true_positives_count',
                                                                            'true_positives_rate',
                                                                            'false_positives_count',
                                                                            'false_positives_rate'])
        return df

    @staticmethod
    def plot(evaluation, plot_type='radar', log_compress=None):

        if type != 'radar':
            raise ValueError('Only "radar" plot type is currently supported')

        def radar_plot(scores_dict, title):
            original_labels = list(scores_dict.keys())
            labels=[]
            for label in original_labels:
                labels.append(label.replace('_', '\n'))
            values = list(scores_dict.values())

            # Number of variables
            num_vars = len(labels)

            # Angles for each axis
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

            # Close the plot (repeat first value)
            values += values[:1]
            angles += angles[:1]

            # Create polar plot
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

            # Draw outline
            ax.plot(angles, values, linewidth=2, zorder=2)
            ax.fill(angles, values, alpha=0.25, zorder=1)

            # Set labels
            ax.set_thetagrids(np.degrees(angles[:-1]), labels)
            ax.tick_params(axis="x", pad=12) #20

            # Value range (0–1)
            ax.set_ylim(0, 1)

            # Optional grid styling
            ax.set_rlabel_position(0)
            ax.yaxis.grid(True)
            ax.xaxis.grid(True)
            ax.set_yticklabels([])

            #ax.set_title(title, y=1.08)
            ax.set_title(title, pad=20)

            #plt.tight_layout()
            plt.show()

        def apply_log_compress(x, alpha=100):
            """
            Compress values toward 1.
            alpha > 0 controls strength (higher = more compression)
            """
            return np.log1p(alpha * x) / np.log1p(alpha)

        for model in evaluation:
            scores = {}
            for item in evaluation[model]:
                if item.endswith('_true_positives_rate'):
                    if log_compress:
                        value = apply_log_compress(evaluation[model][item], alpha=log_compress)
                    else:
                        value = evaluation[model][item]
                    scores[item.replace('_true_positives_rate','')] = value

            radar_plot(scores, title="Model \"{}\"".format(model)) # detection strengths

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

def _variable_granularity_evaluation(flagged_timeseries_df,anomaly_labels_df,breakdown=False):
    one_series_evaluation_result = {}
    flag_columns_n = len(flagged_timeseries_df.filter(like='anomaly').columns)
    variables_n = len(flagged_timeseries_df.columns) - flag_columns_n
    if variables_n != 1 and variables_n != flag_columns_n:
        raise ValueError('Variable granularity is not for this model')
    normalization_factor = variables_n * len(flagged_timeseries_df)

    total_inserted_anomalies_n = 0
    total_detected_anomalies_n = 0
    breakdown_info = {}
    false_positives_count = 0
    for anomaly,frequency in anomaly_labels_df.value_counts(dropna=False).items():
        if anomaly is not None:
            total_inserted_anomalies_n += frequency
        anomaly_count = 0
        for timestamp in flagged_timeseries_df.index:
            if anomaly_labels_df[timestamp] == anomaly:
                for column in flagged_timeseries_df.filter(like='anomaly').columns:
                    if anomaly is not None:
                        anomaly_count += flagged_timeseries_df.loc[timestamp,column]
                    else:
                        false_positives_count += flagged_timeseries_df.loc[timestamp,column]
        if anomaly is not None:
            total_detected_anomalies_n += anomaly_count
            breakdown_info[anomaly + '_true_positives_count'] = anomaly_count
            breakdown_info[anomaly + '_true_positives_rate'] = anomaly_count/(frequency * variables_n)

    total_inserted_anomalies_n *= variables_n
    one_series_evaluation_result['false_positives_count'] = false_positives_count
    one_series_evaluation_result['false_positives_ratio'] = false_positives_count/normalization_factor
    one_series_evaluation_result['true_positives_count'] = total_detected_anomalies_n
    if total_inserted_anomalies_n:
        one_series_evaluation_result['true_positives_rate'] = total_detected_anomalies_n/total_inserted_anomalies_n
    else:
        one_series_evaluation_result['true_positives_rate'] = None
    if breakdown:
        return one_series_evaluation_result | breakdown_info
    else:
        return one_series_evaluation_result

def _point_granularity_evaluation(flagged_timeseries_df,anomaly_labels_df,breakdown=False):
    one_series_evaluation_result = {}
    normalization_factor = len(flagged_timeseries_df)

    total_inserted_anomalies_n = 0
    total_detected_anomalies_n = 0
    breakdown_info = {}
    false_positives_count = 0
    for anomaly,frequency in anomaly_labels_df.value_counts(dropna=False).items():
        if anomaly is not None:
            total_inserted_anomalies_n += frequency
        anomaly_count = 0
        for timestamp in flagged_timeseries_df.index:
            if anomaly_labels_df[timestamp] == anomaly:
                for column in flagged_timeseries_df.filter(like='anomaly').columns:
                    if flagged_timeseries_df.loc[timestamp,column]:
                        if anomaly is not None:
                            anomaly_count += 1
                        else:
                            false_positives_count += 1
                        break
        if anomaly is not None:
            total_detected_anomalies_n += anomaly_count
            breakdown_info[anomaly + '_true_positives_count'] = anomaly_count
            breakdown_info[anomaly + '_true_positives_rate'] = anomaly_count/frequency

    one_series_evaluation_result['false_positives_count'] = false_positives_count
    one_series_evaluation_result['false_positives_ratio'] = false_positives_count/normalization_factor
    one_series_evaluation_result['true_positives_count'] = total_detected_anomalies_n
    if total_inserted_anomalies_n:
        one_series_evaluation_result['true_positives_rate'] = total_detected_anomalies_n/total_inserted_anomalies_n
    else:
        one_series_evaluation_result['true_positives_rate'] = None
    if breakdown:
        return one_series_evaluation_result | breakdown_info
    else:
        return one_series_evaluation_result

def _series_granularity_evaluation(flagged_timeseries_df,anomaly_labels_df,breakdown=False):
    anomalies = []
    for anomaly,frequency in anomaly_labels_df.value_counts(dropna=False).items():
        if anomaly is not None:
            anomalies.append(anomaly)
    if len(anomalies) != 1 and breakdown:
        raise ValueError('Series must have only 1 anomaly type for breakdown in mode granularity = "series"')

    one_series_evaluation_result = {}
    breakdown_info = {}
    is_series_anomalous = 0
    for timestamp in flagged_timeseries_df.index:
        for column in flagged_timeseries_df.filter(like='anomaly').columns:
            if flagged_timeseries_df.loc[timestamp,column]:
                is_series_anomalous = 1
                if anomalies:
                    inserted_anomaly = anomalies[0]
                    breakdown_info[inserted_anomaly + '_true_positives_count'] = 1
                    breakdown_info[inserted_anomaly + '_true_positives_rate'] = 1
                break
    one_series_evaluation_result['false_positives_count'] = 1 if is_series_anomalous and not anomalies else 0
    one_series_evaluation_result['false_positives_ratio'] = one_series_evaluation_result['false_positives_count']
    one_series_evaluation_result['true_positives_count'] = 1 if is_series_anomalous and anomalies else 0
    one_series_evaluation_result['true_positives_rate'] = one_series_evaluation_result['true_positives_count'] if anomalies else None

    if breakdown:
        return one_series_evaluation_result | breakdown_info
    else:
        return one_series_evaluation_result

def _variable_eval_with_events_strategy(sample_df,anomaly_labels_df,breakdown=False):
    raise NotImplementedError('Evaluation with events strategy and variable granularity not implemented')

def _point_eval_with_events_strategy(flagged_timeseries_df,anomaly_labels_df,breakdown=False):
    detected_events_n = 0
    false_positives_n = 0
    events_n, event_type_counts, event_time_slots = _count_anomalous_events(anomaly_labels_df)
    evaluation_result = {}
    breakdown_info = {}

    flags_df = flagged_timeseries_df.filter(like='anomaly')
    previous_point_info = {'label': 0}
    for timestamp in flagged_timeseries_df.index:
        anomaly_label = anomaly_labels_df.loc[timestamp]
        # True if there is at least 1 variable detected as anomalous
        is_anomalous = flags_df.loc[timestamp].any()
        point_info = {anomaly_label: is_anomalous}
        if point_info != previous_point_info and is_anomalous:
            if anomaly_label is None:
                false_positives_n += 1
        previous_point_info = point_info

    evaluation_result['false_positives_count'] = false_positives_n
    evaluation_result['false_positives_ratio'] = false_positives_n/len(anomaly_labels_df)

    for anomaly, anomaly_time_slots in event_time_slots.items():
        anomaly_n = len(anomaly_time_slots)//2
        events_n += anomaly_n
        detected_anomaly_n = 0
        for i in range(0,len(anomaly_time_slots),2):
            start = anomaly_time_slots[i]
            stop = anomaly_time_slots[i+1]
            is_detected = flags_df.loc[start:stop].any().any()
            if is_detected:
                detected_anomaly_n +=1
        breakdown_info[anomaly + 'true_positives_count'] = detected_anomaly_n
        breakdown_info[anomaly + 'true_positives_rate'] = detected_anomaly_n/anomaly_n
        detected_events_n += detected_anomaly_n

    evaluation_result['true_positives_count'] = detected_events_n
    if events_n:
        evaluation_result['true_positives_rate'] = detected_events_n/events_n
    else:
        evaluation_result['true_positives_rate'] = None

    if breakdown:
        return evaluation_result | breakdown_info
    else:
        return evaluation_result

def _series_eval_with_events_strategy(sample_df,anomaly_labels_df,breakdown=False):
    raise NotImplementedError('Evaluation with events strategy and series granularity not implemented')

def _count_anomalous_events(anomaly_labels_df):
    events_n = 0
    event_type_counts = {}
    event_time_slots = {}
    previous_anomaly_label = None
    for timestamp in anomaly_labels_df.index:
        anomaly_label = anomaly_labels_df.loc[timestamp]
        if anomaly_label != previous_anomaly_label:
            if anomaly_label is not None:
                events_n += 1
                # To manage series with adjoining anomalies
                if previous_anomaly_label is not None:
                    event_time_slots[key].append(previous_timestamp)
                key = anomaly_label
                if anomaly_label in event_type_counts.keys():
                    event_type_counts[key] +=1
                    event_time_slots[key].append(timestamp)
                else:
                    event_type_counts[key] =1
                    event_time_slots[key] = [timestamp]

            else:
                event_time_slots[key].append(previous_timestamp)

        previous_timestamp = timestamp
        previous_anomaly_label = anomaly_label
    # To manage series ending with an anomaly
    if previous_anomaly_label is not None:
        event_time_slots[key].append(previous_timestamp)
    return events_n , event_type_counts, event_time_slots
