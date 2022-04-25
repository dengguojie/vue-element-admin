#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Input data range inference rules
"""
# Standard Packages
import logging
from typing import Sequence
from functools import wraps

# Third-Party Packages
from .element_wise import _add
from .element_wise import _sub
from .element_wise import _div
from .element_wise import _mod
from .element_wise import _mul
from .element_wise import _maximum
from .element_wise import _minimum
from .element_wise import _floor_div
from .element_wise import _real_div
from .element_wise import _floor_mod
from .element_wise import _relu_grad_v2
from .elementary_reduce import _reduce_sum
from .elementary_reduce import _reduce_sum_d
from .elementary_reduce import _reduce_prod
from .elementary_reduce import _bias_add_grad
from .elementary_reduce import _reduce_mean_d

input_range_func = {
    "add": _add,
    "sub": _sub,
    "div": _div,
    "mod": _mod,
    "mul": _mul,
    "maximum": _maximum,
    "minimum": _minimum,
    "floor_div": _floor_div,
    "real_div": _real_div,
    "floor_mod": _floor_mod,
    "reduce_sum": _reduce_sum,
    "reduce_sum_d": _reduce_sum_d,
    "reduce_prod": _reduce_prod,
    "bias_add_grad": _bias_add_grad,
    "reduce_mean_d": _reduce_mean_d,
    "relu_grad_v2": _relu_grad_v2
}


def register_input_range(operator_names: Sequence[str]):
    """Register input function"""

    def __inner_input_range_registry(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        for operator_name in operator_names:
            if operator_name in input_range_func:
                logging.warning("input range function of %s has already been registered!" % operator_name)
            input_range_func[operator_name] = __wrapper
        return __wrapper

    return __inner_input_range_registry
