#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Precious utility functions for input range inference
"""
# Third-Party Packages
from ...utilities import get


def input_check(dyn_inputs: tuple, dyn_input_dtypes: tuple, input_num: int, input_groups: tuple = None) -> bool:
    """
    This utility function helps check inputs
    :param dyn_inputs:
    :param dyn_input_dtypes:
    :param input_num:
    :param input_groups:
    :return:
    """
    # Check input num
    if len(dyn_inputs) != input_num or (len(dyn_input_dtypes) != 1 and len(dyn_input_dtypes) != input_num):
        return False
    # Check input dtype
    if input_groups is None:
        # All inputs should have the same dtype
        first_input_dtype = dyn_input_dtypes[0]
        for dtype in dyn_input_dtypes:
            if dtype != first_input_dtype:
                return False
    else:
        for input_group in input_groups:
            first_input_dtype = get(dyn_input_dtypes, input_group[0])
            for idx in input_group:
                dtype = get(dyn_input_dtypes, idx)
                if dtype != first_input_dtype:
                    return False
    return True
