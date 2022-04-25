#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Logging module
"""
# Standard Packages
import sys
import contextlib

# Third-party Packages
import numpy
import warnings
import logging.handlers


class MyFilter(object):
    """
    This Filter is used to make logging module to print only message of specific level
    """

    def __init__(self, level):
        self.__level = level

    def filter(self, record):
        """
        filter for printing only message of specific level
        """
        return record.levelno == self.__level


def attach_handler(_handler, _level=None, _filter=None, _formatter=None):
    """
    :param _handler:
    :param _level:
    :param _formatter:
    :param _filter:
    :return:
    """
    if _level is not None:
        _handler.setLevel(_level)
    if _formatter is not None:
        _handler.setFormatter(_formatter)
    else:
        log_format = '%(asctime)s [%(levelname)s] ' \
                     '[%(process)d %(processName)s %(threadName)s] [%(filename)s:%(lineno)d]: ' \
                     '%(message)s '
        formatter = logging.Formatter(log_format)
        _handler.setFormatter(formatter)
    if _filter is not None:
        _handler.addFilter(_filter)
    logging.getLogger().addHandler(_handler)
    # noinspection PyUnresolvedReferences
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.addHandler(_handler)


def add_level(name: str, visible_name: str, level: int):
    """Add a new logging level"""

    def _logging_method(self, message, *args, **kwargs):
        if self.isEnabledFor(level):
            self._log(level, message, args, **kwargs)

    def _logging_method_root(message, *args, **kwargs):
        logging.log(level, message, *args, **kwargs)

    logging.addLevelName(level, visible_name)
    setattr(logging, name, _logging_method_root)
    setattr(logging.getLoggerClass(), name, _logging_method)


def default_logging_config(file_handler: bool = False, testcase_name: str = None):
    for handler in logging.getLogger().handlers:
        logging.getLogger().removeHandler(handler)
    # noinspection PyUnresolvedReferences
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    # Ignore numpy RuntimeWarnings
    numpy.seterr(all='ignore')
    warnings.filterwarnings('ignore')
    # Attach logging handler for internal logging levels
    if file_handler:
        if not testcase_name:
            attach_handler(logging.FileHandler('tbetoolkits-debug.log'), logging.DEBUG)
            attach_handler(logging.FileHandler('tbetoolkits-info.log'), logging.INFO, MyFilter(logging.INFO))
            attach_handler(logging.FileHandler('tbetoolkits-warning.log'), logging.WARNING, MyFilter(logging.WARNING))
            attach_handler(logging.FileHandler('tbetoolkits-critical.log'), logging.CRITICAL,
                           MyFilter(logging.CRITICAL))
            attach_handler(logging.FileHandler('tbetoolkits-error.log'), logging.ERROR, MyFilter(logging.ERROR))
            attach_handler(logging.FileHandler("tbetoolkits-compare.log"), logging.DEBUG - 5,
                           MyFilter(logging.DEBUG - 5))
        else:
            attach_handler(logging.FileHandler(f'tbetoolkits-debug-{testcase_name}.log'), logging.DEBUG)
            attach_handler(logging.FileHandler(f'tbetoolkits-info-{testcase_name}.log'), logging.INFO,
                           MyFilter(logging.INFO))
            attach_handler(logging.FileHandler(f'tbetoolkits-warning-{testcase_name}.log'), logging.WARNING,
                           MyFilter(logging.WARNING))
            attach_handler(logging.FileHandler(f'tbetoolkits-critical-{testcase_name}.log'), logging.CRITICAL,
                           MyFilter(logging.CRITICAL))
            attach_handler(logging.FileHandler(f'tbetoolkits-error-{testcase_name}.log'), logging.ERROR,
                           MyFilter(logging.ERROR))
            attach_handler(logging.FileHandler(f"tbetoolkits-compare-{testcase_name}.log"), logging.DEBUG - 5,
                           MyFilter(logging.DEBUG - 5))
        attach_handler(logging.StreamHandler(sys.stdout), logging.INFO)
    else:
        attach_handler(logging.StreamHandler(sys.stdout), logging.NOTSET)
    # Enable all logging
    logging.getLogger().setLevel(logging.NOTSET)
    logging.debug(f"TBEToolkits Default Logging System initialized with file handler status {file_handler}")
    if testcase_name:
        logging.debug(f"Trying to log single testcase log for testcase {testcase_name}")
