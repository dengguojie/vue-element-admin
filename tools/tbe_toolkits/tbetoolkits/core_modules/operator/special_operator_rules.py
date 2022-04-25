#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Special input data generation rules"""
# Standard Packages
import logging
from typing import Sequence
from functools import wraps
# Third-Party Packages
import tbetoolkits


special_operator_registry = {}


def register_special_operator(operator_names: Sequence[str]):
    """Register input function"""

    def __inner_special_operator_registry(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        for operator_name in operator_names:
            if operator_name in special_operator_registry:
                logging.warning("special function of operator %s has already been registered!", operator_name)
            special_operator_registry[operator_name] = __wrapper
        return __wrapper

    return __inner_special_operator_registry


@register_special_operator(["relu_grad_v2"])
def _relu_grad_v2(testcase: "tbetoolkits.UniversalTestcaseStructure", mode: str):
    if mode == "dynamic":
        testcase.stc_inputs = list(testcase.stc_inputs)
        testcase.stc_inputs[1] = testcase.stc_inputs[0]
        testcase.stc_input_dtypes = list(testcase.stc_input_dtypes)
        testcase.stc_input_dtypes[1] = "uint1"
