from .anomaly_detectors import MinMaxAnomalyDetector


def evaluate_anomaly_detector(anomaly_detector, evaluation_timeseries_df, synthetic=False):

    if not isinstance(anomaly_detector,MinMaxAnomalyDetector):
        raise ValueError('Only MinMaxAnomalyDetector is supported')

    if 'anomaly_label' not in evaluation_timeseries_df.columns:
        raise ValueError('The anomaly_label column is missing: it is necessary for evaluation')
    # Series with anomaly labels of data pints
    anomaly_labels = evaluation_timeseries_df['anomaly_label']

    if synthetic:
        no_flag_timeseries_df = evaluation_timeseries_df.drop(columns=['time','anomaly_label','effect_label'])
    else:
        no_flag_timeseries_df = evaluation_timeseries_df.drop(columns=['anomaly_label'])

    evaluated_timeseries_df = anomaly_detector.apply(no_flag_timeseries_df)
    evaluated_anomaly_flags = evaluated_timeseries_df.filter(like='_anomaly')
    evaluation_results = {}

    for anomaly_label,frequency in anomaly_labels.value_counts().items():

        for raw in range(len(evaluated_timeseries_df)):

            if anomaly_labels.iloc[raw] == anomaly_label:
                evaluation_results[anomaly_label] = evaluated_anomaly_flags.iloc[[raw]].isin([1])
                break

    return evaluation_results

