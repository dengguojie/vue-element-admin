#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
dilation
"""
from impl.util.platform_adapter import error_manager_util
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm


def _param_check(x, dilations):
    shape_x = x.get("shape")
    if shape_x is None:
        args_dict = {
            "errCode": "E60029",
            "param_name": "x",
            "key": "shape"
        }
        raise RuntimeError(
            args_dict,
            error_manager_util.get_error_message(args_dict)
        )
    shape_x_len = len(shape_x)
    if shape_x_len != len(dilations):
        args_dict = {
            "errCode": "E60002",
            "attr_name": "dim",
            "param1_name": "x",
            "param2_name": "dilations",
            "param1_value": "{}".format(shape_x_len),
            "param2_value": "{}".format(len(dilations))
        }
        raise RuntimeError(
            args_dict,
            error_manager_util.get_error_message(args_dict)
        )
    for value in dilations:
        if not isinstance(value, int) or value <= 0:
            error_manager_cube.raise_err_specific(
                "Dilation",
                "Elements in dilations should be positive integer"
                )
    dtype = x.get("dtype")
    if dtype not in ("int8", "float16", "float32"):
        args_dict = {
            "errCode": "E60005",
            "param_name": "x",
            "expected_dtype": "(int8, float16, float32)",
            "dtype": "{}".format(dtype)
        }
        raise RuntimeError(
            args_dict,
            error_manager_util.get_error_message(args_dict)
        )


def dilation(x, y, dilations, pads=None, padding_value=0.0, kernel_name="dilation"):
    """
    dilation, expand dims of x according to parameter dilations, for example:
    when x = [[1, 1], [1, 1]], dilations = [2, 1], output is [[1, 1], [0, 0], [1, 1]]
    :param x: dict, input
    :param y: dict, output
    :param dilations: list or tuple
    :param padding_value: float
    :param pads: list or tuple or None
    :param kernel_name
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    _param_check(x, dilations)
    attrs = {
        "dilations": dilations,
        "dtype": dtype
    }
    tensor_x = tvm.placeholder(shape, dtype=dtype, name="x", attrs=attrs)
    dilated_x = tbe.dilation(tensor_x, dilations, pads, padding_value)

    with tvm.target.cce():
        s = tbe.auto_schedule(dilated_x)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": (tensor_x, dilated_x)
    }
    tbe.build(s, config)
