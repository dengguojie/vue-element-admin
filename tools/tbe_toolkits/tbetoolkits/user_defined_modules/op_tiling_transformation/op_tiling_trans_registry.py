#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""op_tiling transformation parameters function registry"""
# Standard Packages
import logging
from typing import Sequence
from functools import wraps

op_tiling_trans_map = {}


def register_func(operator_names: Sequence[str]):
    """Register op_tiling_trans function"""
    if not isinstance(operator_names, (list, tuple)):
        raise TypeError("Register function for op_tiling_trans funcs must receive a list or tuple, not %s"
                        % str(operator_names))

    def __inner_func_registry(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        for operator_name in operator_names:
            if operator_name in op_tiling_trans_map:
                logging.warning("op_tiling_trans function of %s has already been registered!" % operator_name)
            op_tiling_trans_map[operator_name] = __wrapper
        return __wrapper

    return __inner_func_registry
