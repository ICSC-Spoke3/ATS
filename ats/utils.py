# -*- coding: utf-8 -*-
"""Utilities"""

import pandas as pd
import numpy as np
import plotly.express as px
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
    
    
def normalize_df(df, parameters_subset=None,save=False):
    """
    Normalizes a single column of a DataFrame using (value-min)/(max-min).

    Args:
        df (pd.DataFrame): Input DataFrame.
        parameters_subset (list, opt): List of column names to normalize. If None, all columns are used.
        save (bool ,opt): save the DataFrame in "normalized_output.csv" file

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
            logger.error(f"Column '{parameter}' is not of a normalizable type.")

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
    try:
        fig = px.scatter_3d( df_plot, x=x, y=y, z=z, color=color, hover_data=hover_columns )
        fig.update_traces(marker=dict(size=marker_size))
        if show:
             fig.show(renderer=renderer)
    except KeyError as ke:
        missing_cols = [col for col in (x, y, z, color) if col not in df_plot.columns]
        logger.error(f"Column(s) {missing_cols} not found in DataFrame.")
    except Exception as e:
        logger.error(e)
    
    return fig

def save_df_to_csv(df, outputfile="normalized_output.csv"):
    """
    Save a DataFrame to CSV,including column headers and excluding the index.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        outputfile (str, optional): The output CSV file path. 
            Defaults to "normalized_output.csv".

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
    try:
        # try renaming
        df.rename(columns={old_name: new_name}, inplace=True)
        logger.info(f" Column '{old_name}' renamed to '{new_name}'.")
    
    except KeyError as ke:
        # the column does not exist
        logger.error(f" Error: the name '{old_name}' does not exist. Available columns: {list(df.columns)}")
    
    except Exception as e:
        # any other error
        logger.error(f" Unable to rename column '{old_name}': {e} (type: {type(e).__name__})")
    
    return df


def merge_df(df1, df2):
    """
    Merge two DataFrames side by side (column-wise).

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame with columns from both inputs.
    """
    return pd.concat([df1, df2], axis=1)


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
        logger.error(f"Mode '{mode}' is not valid. Use one of {list(operations.keys())}.")

    try:
        idx_best = operations[mode]()
        return df.loc[idx_best]
    except KeyError:
        logger.error(f" '{parameter}' does'nt exist. Aviables columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Error finding {mode} for '{parameter}': {e} ({type(e).__name__})")
    
    return df.loc[idx_best]

def plotter_from_df(df, x,y,fixed_parameters):
    """
    Filters a DataFrame based on fixed parameters and plots y vs x.

    Args:
        df (pd.DataFrame): Input DataFrame.
        x (str): Column for the x-axis.
        y (str): Column for the y-axis.
        fixed_parameters (dict): Dictionary of column=value pairs (or list/tuple of values) to filter by.

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
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
 
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_filtered[x], df_filtered[y], marker="o")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x}" + (f" | {context_info}" if context_info else ""))
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        return fig

    except Exception as e:
        logger.error(f"Error while plotting {y} vs {x}: {e}")
        return None