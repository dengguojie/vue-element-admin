#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Special input data generation rules"""
# Standard Packages
import logging
from typing import Sequence
from functools import wraps

special_input_func = {}


def register_input(operator_names: Sequence[str]):
    """Register input function"""

    def __inner_input_registry(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        for operator_name in operator_names:
            if operator_name in special_input_func:
                logging.warning("input function of %s has already been registered!" % operator_name)
            special_input_func[operator_name] = __wrapper
        return __wrapper

    return __inner_input_registry
