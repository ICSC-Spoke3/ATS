from .anomaly_detectors.naive import MinMaxAnomalyDetector
import pandas as pd

def _format_for_anomaly_detector(input_df,synthetic=False):
    if synthetic:
        input_df.set_index(input_df['time'],inplace=True)
        input_df.drop(columns=['time'],inplace=True)
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

