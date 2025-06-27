# -*- coding: utf-8 -*-
"""Library logging setup."""

import os
import logging

LOGLEVEL = os.environ.get('ATS_LOGLEVEL') if os.environ.get('ATS_LOGLEVEL') else 'CRITICAL'

levels_mapping = { 50: 'CRITICAL',
                   40: 'ERROR',
                   30: 'WARNING',
                   20: 'INFO',
                   10: 'DEBUG',
                    0: 'NOTSET'}


def setup(level=LOGLEVEL, force=False):
    """Set up the library logger on a given log level.

        Args:
            level(str): the log level between DEBUG, INFO, WARNING, ERROR, and CRITICAL. Defaults to
                        CRITICAL or the value defined by the ATS_LOGLEVEL environment variable.
            force(bool): if to force the setup, even if the logger is already configured.
    """

    ats_logger = logging.getLogger('ats')
    ats_logger.propagate = False
    try:
        configured = False
        for handler in ats_logger.handlers:
            if handler.get_name() == 'ats_handler':
                configured = True
                if force:
                    handler.setLevel(level=level) # Set global ats logging level
                    ats_logger.setLevel(level=level) # Set global ats logging level
                else:
                    if levels_mapping[handler.level] != level.upper():
                        ats_logger.warning('You tried to setup the logger with level "{}" but it is already configured with level "{}". Use force=True to force reconfiguring it.'.format(level, levels_mapping[handler.level]))
    except IndexError:
        configured=False

    if not configured:
        ats_handler = logging.StreamHandler()
        ats_handler.set_name('ats_handler')
        ats_handler.setLevel(level=level) # Set ats default (and only) handler level
        ats_handler.setFormatter(logging.Formatter('[%(levelname)s] %(name)s: %(message)s'))
        ats_logger.addHandler(ats_handler)
        ats_logger.setLevel(level=level) # Set global ats logging level

