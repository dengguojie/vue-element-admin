#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""davinci to tensorflow parameters function registry"""
# Standard Packages
import logging
from typing import Sequence
from functools import wraps

special_shape_inference_func_map = {}


def register_func(operator_names: Sequence[str]):
    """Register shape inference function"""
    if not isinstance(operator_names, (list, tuple)):
        raise TypeError("Register function for shape inference funcs must receive a list or tuple, not %s"
                        % str(operator_names))

    def __inner_func_registry(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        for name in operator_names:
            if name in special_shape_inference_func_map:
                logging.warning("shape inference function of %s has already been registered!", name)
            special_shape_inference_func_map[name] = __wrapper
        return __wrapper

    return __inner_func_registry