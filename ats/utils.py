# -*- coding: utf-8 -*-
"""Utilities"""

import pandas as pd
import numpy as np
from timeseria.datastructures import TimeSeries

def generate_timeseries_df(start='2025-06-10 14:00:00',  tz='UTC', freq='H', entries=10, pattern='sin', variables=1):
    if pattern not in ['sin']:
        raise ValueError(f'Unknown pattern "{pattern}"')

    time_index = pd.date_range(
        start=pd.Timestamp(start),
        periods=entries,
        freq=freq,
        tz=tz
    )

    data = {}
    for i in range(variables):
        col_name = 'value' if variables == 1 else f'value_{i+1}'
        data[col_name] = np.sin(np.arange(entries) + i * np.pi / 4)

    df = pd.DataFrame(data, index=time_index)
    df.index.name = 'timestamp'
    return df

def plot_timeseries_df(timeseries_df, *args, **kwargs):

    timeseries = TimeSeries.from_df(timeseries_df)

    # Convert anomaly flags in data indexes
    for datapoint in timeseries:
        for data_label in datapoint.data_labels():
            if data_label.endswith('_anomaly'):
                datapoint.data_indexes[data_label] = datapoint.data.pop(data_label)

    return timeseries.plot(*args, **kwargs)


def normalizza_parametro(df, parametro):
    """
    Normalizza una singola colonna di un DataFrame usando min-max.
    Se min = max, restituisce una colonna piena di 1.

    Args:
        df (pd.DataFrame): DataFrame di input.
        parametro (str): Nome della colonna da normalizzare.

    Returns:
        pd.Series: Colonna normalizzata.
    """
    max_parameter = df[parametro].max()
    min_parameter = df[parametro].min()

    if max_parameter == min_parameter:
        return pd.Series(1, index=df.index)
    else:
        return (df[parametro] - min_parameter) / (max_parameter - min_parameter)