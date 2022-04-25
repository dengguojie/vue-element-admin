#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Davinci operator parameter transformation map for pytorch
"""
# Standard Packages

# Third-party Packages
import tbetoolkits
import tbetoolkits.core_modules.testcase_manager.testcase_manager
from . import dav2torch_registry


def extract_params(testcase_struct, param_name: str, target_name: str = None):
    res = None
    if param_name in testcase_struct.other_compilation_params:
        res = testcase_struct.other_compilation_params.get(param_name)
    if param_name in testcase_struct.other_runtime_params:
        res = testcase_struct.other_runtime_params.get(param_name)
    if target_name is not None:
        testcase_struct.other_compilation_params = testcase_struct.other_runtime_params = {target_name: res}
    return res


@dav2torch_registry.register_func(["reduce_sum", "reduce_sum_d",
                                   "reduce_all", "reduce_all_d",
                                   "reduce_any", "reduce_any_d",
                                   "reduce_max", "reduce_max_d",
                                   "reduce_prod", "reduce_prod_d"])
def reduce_torch_transformation(testcase_struct: tbetoolkits.core_modules.testcase_manager.testcase_manager.UniversalTestcaseStructure):
    testcase_struct.op_name = testcase_struct.op_name.replace("reduce_", "").replace("_d", "")
    axis = None
    if "axis" in testcase_struct.other_compilation_params:
        axis = testcase_struct.other_compilation_params["axis"]
    if "axes" in testcase_struct.other_compilation_params:
        axis = testcase_struct.other_compilation_params["axes"]
    if "axis" in testcase_struct.other_runtime_params:
        axis = testcase_struct.other_runtime_params["axis"]
    if "axes" in testcase_struct.other_runtime_params:
        axis = testcase_struct.other_runtime_params["axes"]
    testcase_struct.other_compilation_params = testcase_struct.other_runtime_params = {"dim": axis}
    testcase_struct.torch_func = get_torch_func(testcase_struct.op_name)
    return testcase_struct


@dav2torch_registry.register_func(["fill"])
def fill_torch_transformation(testcase_struct: tbetoolkits.core_modules.testcase_manager.testcase_manager.UniversalTestcaseStructure):
    import numpy
    shape = None
    testcase_struct.op_name = "full"
    dtype = testcase_struct.stc_input_dtypes[0]
    if "dims" in testcase_struct.other_compilation_params:
        shape = testcase_struct.other_compilation_params["dims"]
    if "dims" in testcase_struct.other_runtime_params:
        shape = testcase_struct.other_runtime_params["dims"]
    testcase_struct.other_compilation_params = testcase_struct.other_runtime_params = \
        {"size": shape, "fill_value": getattr(numpy, dtype)(5), "device": None}
    testcase_struct.stc_ori_inputs = []
    testcase_struct.torch_func = get_torch_func(testcase_struct.op_name)
    return testcase_struct


@dav2torch_registry.register_func(["tile_d", "tile"])
def tile_torch_transformation(testcase_struct: tbetoolkits.core_modules.testcase_manager.testcase_manager.UniversalTestcaseStructure):
    testcase_struct.op_name = "tile"
    extract_params(testcase_struct, "multiples", "dims")
    testcase_struct.torch_func = get_torch_func(testcase_struct.op_name)
    return testcase_struct


@dav2torch_registry.register_func(["floor_mod"])
def floor_mod_torch_transformation(testcase_struct: tbetoolkits.core_modules.testcase_manager.testcase_manager.UniversalTestcaseStructure):
    testcase_struct.op_name = "fmod"
    testcase_struct.torch_func = get_torch_func(testcase_struct.op_name)
    return testcase_struct


@dav2torch_registry.register_func(["equal"])
def equal_torch_transformation(testcase_struct: tbetoolkits.core_modules.testcase_manager.testcase_manager.UniversalTestcaseStructure):
    testcase_struct.op_name = "eq"
    testcase_struct.torch_func = get_torch_func(testcase_struct.op_name)
    return testcase_struct


@dav2torch_registry.register_func(["as_strided"])
def as_strided_torch_transformation(testcase_struct: tbetoolkits.core_modules.testcase_manager.testcase_manager.UniversalTestcaseStructure):
    if "storage_offset" in testcase_struct.other_compilation_params:
        if not isinstance(testcase_struct.other_compilation_params["storage_offset"], int):
            testcase_struct.other_compilation_params["storage_offset"] = \
                testcase_struct.other_compilation_params["storage_offset"][0]
    if "storage_offset" in testcase_struct.other_runtime_params:
        if not isinstance(testcase_struct.other_runtime_params["storage_offset"], int):
            testcase_struct.other_runtime_params["storage_offset"] = \
                testcase_struct.other_runtime_params["storage_offset"][0]

    def exec_as_strided(input_tensor, size, stride, storage_offset=0):
        intermediate_result = get_torch_func(testcase_struct.op_name)(input_tensor, size, stride, storage_offset)
        return intermediate_result.contiguous()
    testcase_struct.torch_func = exec_as_strided
    return testcase_struct


@dav2torch_registry.register_func(["transpose"])
def transpose_torch_transformation(testcase_struct: tbetoolkits.core_modules.testcase_manager.testcase_manager.UniversalTestcaseStructure):
    testcase_struct.op_name = "permute"
    extract_params(testcase_struct, "perm", "dims")

    def contiguous_transpose(*args, **kwargs):
        return get_torch_func(testcase_struct.op_name)(*args, **kwargs).contiguous()
    testcase_struct.torch_func = contiguous_transpose

    return testcase_struct


# noinspection PyUnresolvedReferences
def get_torch_func(_op_name: str):
    """
    Get pytorch function by op_name
    """
    import torch
    return getattr(torch, _op_name, None)
