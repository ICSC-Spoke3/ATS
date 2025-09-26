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


def normalize_parameter(df, parameter):
    """
    Normalizes a single column of a DataFrame using (value-min)/(max-min).
    If min = max, it returns a column filled with 1s.

    Args:
        df (pd.DataFrame): Input DataFrame.
        parametro (str): Name of the column to normalize.

    Returns:
        pd.Series: Normalized column.
    """
    max_parameter = df[parameter].max()
    min_parameter = df[parameter].min()

    if max_parameter == min_parameter:
        return pd.Series(1, index=df.index, name = parameter)
    else:
        return (df[parameter] - min_parameter) / (max_parameter - min_parameter)
    
    
def normalize_df(df, parameters_subset=None):
    """
    Normalizes a single column of a DataFrame using (value-min)/(max-min).

    Args:
        df (pd.DataFrame): Input DataFrame.
        parameters_subset (list, opt): List of column names to normalize. If None, all columns are used. 

    Returns:
        pd.DataFrame: DataFrame with normalized columns.
    """
    df_norm = pd.DataFrame()

    if parameters_subset:
        parameters = parameters_subset
    else:
        # usa tutte le colonne del DataFrame
        parameters = df.columns

    for parameter in parameters:
        try:
            # prova a leggere il primo valore
            first_val = df[parameter].iloc[0]

            # se è bool, consideriamo tutta la colonna bool → skip
            if isinstance(first_val, (bool, np.bool_)):
                raise TypeError(f"Colonna '{parameter}' rilevata come bool: saltata.")

            # normalizza la colonna
            df_norm[f"{parameter}_norm"] = normalize_parameter(df, parameter)

        except TypeError as te:
            # gestione esplicita dei bool
            print(f"{te}")

        except Exception as e:
            # gestione di altri errori (non numerico, valori non validi, ecc.)
            print(f" Colonna '{parameter}' non può essere normalizzata: {e} (tipo: {type(e).__name__})")

    return df_norm