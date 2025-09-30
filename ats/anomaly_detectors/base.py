# -*- coding: utf-8 -*-
"""Anomaly detectors"""

import pandas as pd

# Setup logging
import logging
logger = logging.getLogger(__name__)


class AnomalyDetector():

    def apply(self, df):
        raise NotImplementedError()

