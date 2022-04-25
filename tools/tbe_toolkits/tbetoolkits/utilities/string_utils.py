#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Precious Synchronization Related Utilities
"""
# Standard Packages
import threading
import multiprocessing
from typing import Any


def tostr(value: Any) -> str:
    """
    Convert objects to meaningful string
    :param value: Anything
    :return:
    """
    result = ""
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return '_'.join(tuple(tostr((key, tostr(value[key]))) for key in value))
    if isinstance(value, (tuple, list)):
        for sub_value in value:
            if isinstance(sub_value, str):
                result += sub_value.replace("-", "neg").replace(".", "p")
            elif isinstance(sub_value, int):
                result += str(sub_value).replace("-", "neg").replace(".", "p")
            elif isinstance(sub_value, (tuple, list)):
                first_process = '__' + '_'.join(tuple(map(str, sub_value))).replace("-", "neg").replace(".", "p")
                result += first_process.replace('(', '').replace(')', '').replace(' ', '').replace(',', '_')
            else:
                raise TypeError('Invalid type %s of %s for string conversion!' % (type(sub_value), str(value)))
    else:
        raise TypeError('Invalid type %s of %s for string conversion!' % (type(value), str(value)))
    return result


def set_process_name(name: str = "MainProcess"):
    """Set process name for logging"""
    multiprocessing.current_process().name = name


def get_process_name() -> str:
    """Set process name for logging"""
    return multiprocessing.current_process().name


def set_thread_name(name: str = "MainThread"):
    """Set current thread name for threading"""
    threading.current_thread().setName(name)
