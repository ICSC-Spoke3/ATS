# -*- coding: utf-8 -*-
"""Utilities"""

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from timeseria.datastructures import TimeSeries

# Setup logging
import logging
logger = logging.getLogger(__name__)


def generate_timeseries_df(start='2025-06-10 14:00:00',  tz='UTC', freq='h', entries=10, pattern='sin', variables=1):
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
        parameter (str): Name of the column to normalize.

    Returns:
        pd.Series: Normalized column.
    """
    max_parameter = df[parameter].max()
    min_parameter = df[parameter].min()

    if max_parameter == min_parameter:
        return pd.Series(1.0, index=df.index, name=parameter)
    else:
        return (df[parameter] - min_parameter) / (max_parameter - min_parameter).astype(float)
    
def normalize_df(df, parameters_subset=None,save=False):
    """
    Normalizes a DataFrame using (value-min)/(max-min).

    Args:
        df (pd.DataFrame): Input DataFrame.
        parameters_subset (list, opt): List of column names to normalize. If None, all columns are used.
        save (bool, opt): save the DataFrame in "normalized_output.csv" file

    Returns:
        pd.DataFrame: DataFrame with normalized columns.
    """
    df_norm = pd.DataFrame()

    if parameters_subset:
        parameters = parameters_subset
    else:
        parameters = df.columns

    for parameter in parameters:
        try:
            df_norm[f"{parameter}_norm"] = normalize_parameter(df, parameter)
            logger.debug(f"Column '{parameter}' normalized successfully.")

        except TypeError as te:
            logger.error(f"Column '{parameter}' is not of a normalizable type. '{parameter}' ignored")

        except Exception as e:
            logger.error(f"Normalization failed for column '{parameter}': {e} (type: {type(e).__name__}).")

    if save:
        save_df_to_csv(df_norm, outputfile="normalized_output.csv")

    return df_norm


def plot_3d_interactive(df,x="avg_err",y="max_err",z="ks_pvalue",color="fitness",filters=None,
                        hover_columns=None,marker_size=3,renderer="notebook",show = True):
    """
    Creates an interactive 3D scatter plot with Plotly, with optional filters on the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        x, y, z (str): Columns for the axes.
        color (str): Column used to color the points.
        filters (dict): Dictionary with key=column, value=tuple(min,max) to filter the data.
        hover_columns (list): Columns to display on hover. Default: all columns.
        marker_size (int): Size of the markers.
        renderer (str): Plotly renderer ("notebook", "browser", etc.)
        show (bool): Whether to display the figure immediately.

    Returns:
        plotly.graph_objs._figure.Figure: Interactive figure.
    """

    df_plot = df.copy()
    fig = None 
    allowed_filter_cols = {x, y, z, color}

    # Apply filters if provided
    if filters:
        for col, (min_val, max_val) in filters.items():
            if col not in  allowed_filter_cols:
                logger.warning(f"Column '{col}' is not used in the plot (x, y, z, color). Filter ignored.")
            else:
                df_plot = df_plot[(df_plot[col] >= min_val) & (df_plot[col] <= max_val)]

    # If hover_columns not specified, show all columns
    if hover_columns is None:
        hover_columns = df_plot.columns.tolist()

    # Create 3D plot
    missing_cols = [col for col in (x, y, z, color) if col not in df_plot.columns]
    if missing_cols:
        raise KeyError(f"Column(s) {missing_cols} not found in DataFrame.") from None
    try:
        fig = px.scatter_3d(df_plot, x=x, y=y, z=z, color=color, hover_data=hover_columns)
        fig.update_traces(marker=dict(size=marker_size))
        fig.update_layout(width=1000, height=800)
        if show:
            fig.show(renderer=renderer)
        else:
            return fig
    except KeyError as ke:
        logger.error("Missing column when creating 3D scatter: %s", ke)
        return None
    except Exception as e:
        logger.error("Unexpected error while creating 3D interactive plot: %s", e)
        return None     

def save_df_to_csv(df, outputfile="output.csv"):
    """
    Save a DataFrame to CSV,including column headers and excluding the index.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        outputfile (str, optional): The output CSV file path. 
            Defaults to "output.csv".

    Returns:
        None
    """
    df.to_csv(outputfile, index=False, header=True)
    logger.info(f" Saved: {outputfile}")


def rename_column(df, old_name, new_name):
    """
    Renames a column in a DataFrame in Place.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to rename.
        old_name (str): The current column name.
        new_name (str): The new column name.

    Returns:
        pd.DataFrame: The updated DataFrame (renamed in place).
    """
    if old_name not in df.columns:
        raise KeyError(f"Error: column '{old_name}' not found. Available columns: {list(df.columns)}")
    try:
        df.rename(columns={old_name: new_name}, inplace=True)
        logger.info(f" Column '{old_name}' renamed to '{new_name}'.")
    except Exception as e:
        logger.error(f"Unable to rename column '{old_name}': {e} (type: {type(e).__name__})")
        raise
    return df


def merge_df(df1, df2):
    """
    Merge two DataFrames side by side (column-wise) with duplicate column handling.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame with columns from both inputs.
    """
    df2_copy = df2.copy()
    for col in df1.columns.intersection(df2.columns):
        if not df1[col].equals(df2[col]):
            # Rename the column in df2 to avoid conflict
            new_col_name = col + "_df2"
            logger.warning(f"Warning: Column '{col}' has different values in df2. Renaming second DataFrame column to '{new_col_name}'.")
            rename_column(df2_copy,col, new_col_name)
        else:
            # Drop the duplicate column in df2 if identical
            df2_copy.drop(columns=[col], inplace=True)

    return pd.concat([df1, df2_copy], axis=1)


def find_best_parameter(df, parameter, mode="min"):
    """
    Find the row in a DataFrame that has the best (minimum or maximum) value 
    for a given parameter.

    Args:
        df (pd.DataFrame): The DataFrame to search.
        parameter (str): The name of the column to evaluate.
        mode (str, optional): The optimization criterion. 
            Accepts "min" or "max". Defaults to "min".

    Returns:
        pd.Series: The row of the DataFrame corresponding to the best parameter value.
    """
    operations = {
        "min": df[parameter].idxmin,
        "max": df[parameter].idxmax,
        # Others implementation here ...
    }

    if mode not in operations:
        raise ValueError(f"Mode '{mode}' is not valid. Use one of {list(operations.keys())}.")

    try:
        idx_best = operations[mode]()
        return df.loc[idx_best]
    except KeyError:
        logger.error(f" '{parameter}' does'nt exist. Available columns: {list(df.columns)}")
        return None
    except Exception as e:
        logger.error(f"Error finding {mode} for '{parameter}': {e} ({type(e).__name__})")
        return None

def plot_from_df(df, x,y,fixed_parameters=None):
    """
    2D plot of DataFrame (y vs x). It allow to select fixed parameter

    Args:
        df (pd.DataFrame): Input DataFrame.
        x (str): Column for the x-axis.
        y (str): Column for the y-axis.
        fixed_parameters (dict, opt): Dictionary of column=value pairs (or list/tuple of values) to filter by.

    Returns:
         Plot the generated matplotlib figure.
    """
    df_filtered = df.copy()
    if fixed_parameters:
        for key, val in fixed_parameters.items():
            if key not in df_filtered.columns:
                logger.warning(f"'{key}' not in DataFrame columns. Skipping filter.")
                continue
            
            if isinstance(val, (list, tuple, set)):
                df_filtered = df_filtered[df_filtered[key].isin(val)]
            else:
                df_filtered = df_filtered[df_filtered[key] == val]
        
    df_filtered = df_filtered.sort_values(by=[x, y])

    context_info = " | ".join(f"{k}={v}" for k, v in (fixed_parameters or {}).items())
 
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_filtered[x], df_filtered[y], marker="o")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x}" + (f" | {context_info}" if context_info else ""))
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Error while plotting {y} vs {x}: {e}")
        return None