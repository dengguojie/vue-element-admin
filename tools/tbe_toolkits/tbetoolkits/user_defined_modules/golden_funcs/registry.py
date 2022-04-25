#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Golden data generator functions
"""

# Standard Packages
import logging
from typing import Sequence
from functools import wraps

# Third-party Packages
import numpy

golden_funcs = {
    "floor_div": numpy.floor_divide,
    "real_div": numpy.true_divide,
    "neg": numpy.negative,
    "div": numpy.divide,
    "sub": numpy.subtract,
    "mul": numpy.multiply,
    "acos": numpy.arccos,
    "acosh": numpy.arccosh,
    "asin": numpy.arcsin,
    "asinh": numpy.arcsinh,
    "atan": numpy.arctan,
    "atan2": numpy.arctan2,
    "atanh": numpy.arctanh}


def register_golden(operator_names: Sequence[str]):
    """Register golden function"""
    if not isinstance(operator_names, (list, tuple)):
        raise TypeError("Register function for golden funcs must receive a list or tuple, not %s"
                        % str(operator_names))

    def __inner_golden_registry(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        for operator_name in operator_names:
            if operator_name in golden_funcs:
                logging.warning("golden function of %s has already been registered!" % operator_name)
            golden_funcs[operator_name] = __wrapper
        return __wrapper

    return __inner_golden_registry
