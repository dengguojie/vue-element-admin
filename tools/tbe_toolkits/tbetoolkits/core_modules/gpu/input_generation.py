#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Input generation method for GPU Universal testcases
"""
# Third-party Packages
import numpy
from ...utilities import get
from ...utilities import shape_product
from ..testcase_manager import UniversalTestcaseStructure


def __gen_input(context: UniversalTestcaseStructure):
    from ...user_defined_modules.input_funcs import registry as input_registry
    context.input_arrays = []
    for i in range(len(context.stc_ori_inputs)):
        low = get(context.input_data_ranges, i)[0]
        high = get(context.input_data_ranges, i)[1]
        if low is None:
            low = 2
        if high is None:
            high = 100
        size = shape_product(context.stc_ori_inputs[i])
        context.input_arrays.append(numpy.random.uniform(low=low, high=high,
                                                         size=size).reshape(context.stc_ori_inputs[i]).astype(
            get(context.stc_input_dtypes, i)))
    if context.op_name in input_registry.special_input_func:
        context.input_arrays = input_registry.special_input_func[context.op_name](context)[1]
