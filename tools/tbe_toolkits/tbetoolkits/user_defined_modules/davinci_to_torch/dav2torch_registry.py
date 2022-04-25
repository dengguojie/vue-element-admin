#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""davinci to tensorflow parameters function registry"""
# Standard Packages
import logging
from typing import Sequence
from functools import wraps

dav_op_to_torch_map = {}


def register_func(operator_names: Sequence[str]):
    """Register dav2torch function"""
    if not isinstance(operator_names, (list, tuple)):
        raise TypeError("Register function for dav2torch funcs must receive a list or tuple, not %s"
                        % str(operator_names))

    def __inner_func_registry(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        for operator_name in operator_names:
            if operator_name in dav_op_to_torch_map:
                logging.warning("dav2torch function of %s has already been registered!" % operator_name)
            dav_op_to_torch_map[operator_name] = __wrapper
        return __wrapper

    return __inner_func_registry
