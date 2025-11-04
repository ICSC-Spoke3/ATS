import numpy as np
import pandas as pd
from .support_functions import detect_outliers_on_data

# Setup logging
import logging
logger = logging.getLogger(__name__)


class _COMNHARAnomalyDetector:
    """
    Statistically robust anomaly detector based on COM, HAR, and NHAR methodologies.
    """

    def __init__(self, fq=2 * np.pi / 30, fw=2 * np.pi / 7, trend=2, methods=('COM', 'HAR', 'NHAR')):
        self.fq = fq
        self.fw = fw
        self.trend = trend
        self.methods = methods


    def apply(self, data, *args, **kwargs):
        """
        Apply statistical anomaly detection on time series data.

        Args:
            data (pd.DataFrame or list[pd.DataFrame]): 
                Time series data. Each DataFrame must have a DatetimeIndex named 'timestamp',
                and each column represents a variable.

        Returns:
            pd.DataFrame or list[pd.DataFrame]:
                The same structure as input, with added column 'anomaly' (boolean flag).
        """

        if not isinstance(data, pd.DataFrame):
            raise NotImplementedError('This anomaly detector can work only on a single time series (as a Pandas DataFrame)')

        # Sanity checks in input data and  shortcut
        if isinstance(data, pd.DataFrame):
            dataframes = [data]
        elif isinstance(data, list) and all(isinstance(d, pd.DataFrame) for d in data):
            dataframes = data
        else:
            raise TypeError("data must be a pandas DataFrame or list of DataFrames")

        for dataframe in dataframes:
            if not isinstance(dataframe.index, pd.DatetimeIndex):
                raise ValueError("The DataFrame index must be a pandas.DatetimeIndex.")
            if dataframe.index.name != "timestamp":
                raise ValueError("The DataFrame index must be named 'timestamp'.")
            if dataframe.isnull().values.any():
                raise ValueError("Input DataFrame contains missing values. Handle them before applying the detector.")

        # Apply the anomaly detection logic
        results = []
        for dataframe in dataframes:

            # Transpose the dataframe to comply with the detect_outliers_on_data() function requirements
            mY = dataframe.values.T
            detections = detect_outliers_on_data(
                mY,
                fq=self.fq,
                fw=self.fw,
                TREND=self.trend,
                COM='COM' in self.methods,
                HAR='HAR' in self.methods,
                NHAR='NHAR' in self.methods,
            )

            #print(detections['COM'])
            #     row  col
            # 0   80    0
            # 1   81    0
            # 2   81    1
            # 3   80    1
            # 4   80    2
            # 5   80    3

            # Prepare the results dataframe
            results_dataframe = dataframe.copy()
            results_dataframe["anomaly"] = False
            #for variable_name in dataframe.columns:
            #    results_dataframe[f"anomaly_{variable_name}"] = False

            #print(results_dataframe)
            #                        series_0    series_1    series_2    series_3    series_4    series_5    series_6     series_7    series_8    series_9  anomaly
            # timestamp
            # 1970-01-01 00:00:00 -826.248384  119.191281 -249.365737  389.452643   50.196639  170.312065  320.551786   616.361384  -30.979346  134.403614    False
            # 1970-01-01 01:00:00 -898.996784  158.918528 -259.542106  424.651601   61.394984  156.864934  461.685777   604.330570   90.047185 -167.345680    False
            # 1970-01-01 02:00:00 -797.752803  217.121984 -319.006536  245.445180   83.115837   -5.996824  500.712038   704.874015  163.293523 -237.490292    False
            # 1970-01-01 03:00:00 -917.821722  348.643859 -356.400411  218.770569  102.089101   26.065172  588.226671   789.646065  197.049434 -252.260697    False
            # 1970-01-01 04:00:00 -808.093694  225.767563 -373.759195  124.282097   62.543076 -160.780466  548.680429   634.217132  236.499563 -206.523707    False
            # ...                         ...         ...         ...         ...         ...         ...         ...          ...         ...         ...      ...
            # 1970-01-07 01:00:00 -913.614855  225.199078  -77.997378  830.474791  110.799159  781.126422 -166.960595   839.624156  -21.662904  402.938700    False
            # 1970-01-07 02:00:00 -863.794907  424.582153  -49.426454  746.075487  161.494924  723.790468 -197.300783  1053.231844  128.978276  286.985872    False
            # 1970-01-07 03:00:00 -850.261538  396.107547 -206.788703  756.143127   20.729189  548.648614 -142.699670   824.962722  159.918083  264.529751    False
            # 1970-01-07 04:00:00 -952.579415  438.861223 -188.840125  658.077589  122.635578  673.923192   44.354939   961.068127   52.144830  127.281647    False
            # 1970-01-07 05:00:00 -876.011437  237.804984 -231.658239  646.145330   94.996464  461.577686   28.805264   847.240457  121.227161   89.464851    False

            # Assemble the results dataframe
            for method, method_results in detections.items():
                for _ , method_result in method_results.iterrows():

                    variable_index = method_result["col"]
                    variable_name = dataframe.columns[variable_index]
                    time_index = method_result["row"]
                    timestamp = dataframe.index[time_index]

                    logger.debug('Detected anomaly with %s method for variable_index: %s (%s); time_index; %s, (%s)', method, variable_index, variable_name, time_index, timestamp)

                    results_dataframe.loc[timestamp, "anomaly"] = True
                    #results_dataframe.loc[timestamp, f"anomaly_{variable_name}"] = True

            results.append(results_dataframe)

        # Return according to original input data
        return results[0] if isinstance(data, pd.DataFrame) else results



class COMAnomalyDetector(_COMNHARAnomalyDetector):
    """
    Statistically robust anomaly detector based on COM methodology.
    """
    def __init__(self, fq=2 * np.pi / 30, fw=2 * np.pi / 7, trend=2):
        super().__init__(self, fq=fq, fw=fw, trend=trend, methods=('COM'))


class HARAnomalyDetector(_COMNHARAnomalyDetector):
    """
    Statistically robust anomaly detector based on HAR methodology.
    """
    def __init__(self, fq=2 * np.pi / 30, fw=2 * np.pi / 7, trend=2):
        super().__init__(self, fq=fq, fw=fw, trend=trend, methods=('HAR'))


class NHARAnomalyDetector(_COMNHARAnomalyDetector):
    """
    Statistically robust anomaly detector based on NHAR methodology.
    """
    def __init__(self, fq=2 * np.pi / 30, fw=2 * np.pi / 7, trend=2):
        super().__init__(self, fq=fq, fw=fw, trend=trend, methods=('NHAR'))



