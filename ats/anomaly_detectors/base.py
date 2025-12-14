# -*- coding: utf-8 -*-
"""Anomaly detectors"""

import copy
import functools
import pandas as pd
from ..utils import convert_timeseries_df_to_timeseries, convert_timeseries_to_timeseries_df

# Setup logging
import logging
logger = logging.getLogger(__name__)


class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        return self.func(cls)


class AnomalyDetector():

    allowed_capabilities = {
        'mode': {'unsupervised', 'semi-supervised', 'weakly-supervised', 'supervised'},
        'streaming': {True, False},
        'context': {'point', 'window', 'series', 'dataset'},
        'granularity': {'series', 'point', 'variable'},
        'multivariate': {True, False, 'only'},
        'scope': {'specific', 'agnostic'},
    }

    @classproperty
    def capabilities(cls):
        raise NotImplementedError(f'Capabilities are not set for {cls.__name__}')

    def __new__(cls, *args, **kwargs):
        cls._validate_capabilities(cls.capabilities, cls.allowed_capabilities)
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def _validate_capabilities(cls, capabilities, allowed_capabilities):
        missing = set(allowed_capabilities) - set(capabilities)
        if missing:
            raise ValueError(f'Missing required capabilities: {sorted(missing)} for {cls.__name__}')
        for key, value in capabilities.items():
            if key not in allowed_capabilities:
                raise ValueError(f'Unknown capability: "{key}" for {cls.__name__}')
            if value not in allowed_capabilities[key]:
                raise ValueError(f'Invalid value "{value}" for capability "{key}" for {cls.__name__}')


    #========================
    #  Helpers
    #========================
    def set_apply_params(self, **kwargs):
        """
        Set parameters for the apply method of the timeseria model.
        """
        self.apply_params = kwargs

    def get_apply_params(self):
        """
        Get parameters for the apply method of the timeseria model.
        """
        try:
            return self.apply_params
        except:
            return {}

    def set_fit_params(self, **kwargs):
        """
        Set parameters for the fit method of the timeseria model.
        """  
        self.fit_params = kwargs

    def get_fit_params(self):
        """
        Get parameters for the fit method of the timeseria model.
        """  
        try:
            return self.fit_params
        except:
            return {}


    #========================
    #  Decorators
    #========================

    @staticmethod
    def fit_method(fit_method):
        """:meta private:"""
        @functools.wraps(fit_method)
        def do_fit(self, data, **kwargs):

            # Set fit parameters
            fit_params = self.get_fit_params()
            fit_params.update(kwargs)

            # Call fit logic
            fit_method(self, data, **fit_params)

            # Mark as fitted
            self.fitted = True
        return do_fit

    @staticmethod
    def apply_method(apply_method):
        """:meta private:"""
        @functools.wraps(apply_method)
        def do_apply(self, data, **kwargs):

            # Set apply parameters
            apply_params = self.get_apply_params()
            apply_params.update(kwargs)

            # Should we auto-fit?
            if not hasattr(self, 'fitted') or hasattr(self, 'fitted') and not self.fitted:
                # Model is not fitted, try to auto-fit if supported
                try:
                    apply_model = copy.deepcopy(self)
                    apply_model.fit(data, **self.get_fit_params())
                except NotImplementedError:
                    apply_model = self
                else:
                    logger.info('Auto-fitted the anomaly detector')
            else:
                apply_model = self

            return apply_method(apply_model, data, **apply_params)

        return do_apply


    #========================
    #  Interface
    #========================

    def apply(self, data, *args, **kwargs):

        """
        Apply the anomaly detector on some (time series) data.

        Args:
            data (pd.DataFrame or list[pd.DataFrame]): A single time series (in pandas DataFrame format) or a list of time series (in pandas DataFrame format).
            The index of the data frame(s) must be named "timestamp", and each column is supposed to represents a variable.

        Returns:
            pd.DataFrame or list[pd.DataFrame]: the input data with the "anomaly" flag and optional "anomaly_score" columns added on the time series.
        """
        raise NotImplementedError()


    def fit(self, data, *args, **kwargs):
        """
        Fit the anomaly detector on some (time series) data.

        Args:
            data (pd.DataFrame or list[pd.DataFrame]): A single time series (as a pandas DataFrame) or a list of time series (each as a pandas DataFrame).
            The index of the each DataFrame must be named "timestamp", and each column should represents a variable. 
        """
        raise NotImplementedError()



class TimeseriaAnomalyDetector(AnomalyDetector):
    """
    Base class for anomaly detectors wrapped from the timeseria library.
    """

    model_class = None

    def __init__(self, *args, **kwargs):
        if self.model_class is None:
            raise NotImplementedError('Subclasses must define a timeseria model')
        self.model = self.model_class(*args, **kwargs)

    @AnomalyDetector.fit_method
    def fit(self, data, **kwargs):
        """
        Fit the timeseria anomaly detector model.
        """
        if not isinstance(data,pd.DataFrame):
            raise NotImplementedError('Not yet implemented for non DataFrame inputs')
        timeseries_df = data

        # Using timeseria to fit the model
        timeseries = convert_timeseries_df_to_timeseries(timeseries_df)
        self.model.fit(timeseries, **kwargs)

    @AnomalyDetector.apply_method
    def apply(self, data, **kwargs):
        """
        Apply the timeseria anomaly detector model.
        """
        if not isinstance(data,pd.DataFrame):
            raise NotImplementedError('Not yet implemented for non DataFrame inputs')
        timeseries_df = data

        # Using timeseria to fit and apply the model
        timeseries = convert_timeseries_df_to_timeseries(timeseries_df)
        timeseries = self.model.apply(timeseries, **kwargs)

        # Convert back to DataFrame
        timeseries_df = convert_timeseries_to_timeseries_df(timeseries)
        timeseries_df['anomaly'] = (timeseries_df['anomaly'].astype(float) != 0).astype(int)

        return timeseries_df

