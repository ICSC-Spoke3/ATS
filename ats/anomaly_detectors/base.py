# -*- coding: utf-8 -*-
"""Anomaly detectors"""

import copy
import functools
import pandas as pd
from ..utils import convert_timeseries_df_to_timeseries, convert_timeseries_to_timeseries_df

# Setup logging
import logging
logger = logging.getLogger(__name__)


# ---------------------------
# Taxonomy specification
# ---------------------------

ALLOWED_CAPABILITIES = {
    'training': {
        'mode': {
            None,
            'unsupervised',
            'semi-supervised',
            'weakly-supervised',
            'supervised',
        },
        'update': {
            True,
            False
        },
    },
    'inference': {
        'streaming': {
            True,
            False
        },
        'dependency': {
            'point',
            'window',
            'series',
        },
        'granularity': {
            'point',
            'point-labels',
            'series',
            'series-labels',
        },
    },
    'data': {
        'dimensionality': {
            'univariate-single',
            'multivariate-single',
            'univariate-multi',
            'multivariate-multi',
        },
        'sampling': {
            'regular',
            'irregular',
        },
    },
}


# ---------------------------
# Validation logic
# ---------------------------

def validate_capabilities(capabilities):
    '''
    Validate a model capability dictionary.

    Raises
    ------
    ValueError
        If the capability specification is invalid.
    '''
    _validate_capability_structure(capabilities)
    _validate_capability_values(capabilities)
    _validate_capability_cross_constraints(capabilities)


# ---------------------------
# Helpers
# ---------------------------

def _validate_capability_structure(capabilities):
    for section in ALLOWED_CAPABILITIES:
        if section not in capabilities:
            raise ValueError(f"Missing required capability section: '{section}'")

        if not isinstance(capabilities[section], dict):
            raise ValueError(f"Capability section '{section}' must be a dictionary")

        for key in ALLOWED_CAPABILITIES[section]:
            if key not in capabilities[section]:
                raise ValueError(
                    f"Missing required capability section key '{section}.{key}'"
                )


def _validate_capability_values(capabilities):
    for section, fields in ALLOWED_CAPABILITIES.items():
        for field, allowed in fields.items():
            value = capabilities[section][field]

            if isinstance(value, (list, set, tuple)):
                invalid = set(value) - allowed
                if invalid:
                    raise ValueError(
                        f"Invalid value(s) for capability section key '{section}.{field}': {invalid}. "
                        f"Allowed: {allowed}"
                    )
            else:
                if value not in allowed:
                    raise ValueError(
                        f"Invalid value for capability section key '{section}.{field}': '{value}'. "
                        f"Allowed: {allowed}"
                    )


def _validate_capability_cross_constraints(capabilities):
    training_mode = capabilities['training']['mode']
    training_update = capabilities['training']['update']
    inference_streaming = capabilities['inference']['streaming']
    inference_dependency = capabilities['inference']['dependency']
    granularity = capabilities['inference']['granularity']

    # Training constraints
    if training_mode is None and training_update is not False:
        raise ValueError(
            'Capability training update must be False when training mode is None'
        )

    # Streaming constraints
    if inference_streaming is True:
        if inference_dependency in {'series-single', 'series-multi'}:
            raise ValueError(
                'Capability inference streaming cannot require full series dependency'
            )

    # Granularity vs context
    if granularity == 'series' and inference_dependency in {'point', 'window'}:
        raise ValueError(
            'Capability inference granularity for series-level is incompatible with point/window inference dependency'
        )

class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        return self.func(cls)


class AnomalyDetector():

    @classproperty
    def capabilities(cls):
        try:
            cls._capabilities
        except AttributeError:
            raise NotImplementedError(f'Capabilities are not set for {cls.__name__}') from None
        else:
            validate_capabilities(cls._capabilities)
            return cls._capabilities


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
        Apply the anomaly detector on some (time series) data, the the minimum amount of information being the inference dependency.

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

