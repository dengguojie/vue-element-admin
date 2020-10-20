#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
div
"""

import te.lang.cce as tbe
from te import tvm
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from te.utils import para_check
from te.utils import shape_util


# pylint: disable=locally-disabled,too-many-locals,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("div")
def div_compute(input_x, input_y, output_div, kernel_name="div"):
    """
    div compute
    calculating data's div, res =x / y

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_div: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "div"

    Returns
    -------
    res: TVM tensor
        the result of div compute
    """
    input_data1 = shape_util.shape_to_list(input_x.shape)
    input_data2 = shape_util.shape_to_list(input_y.shape)
    shape_list = shape_util.broadcast_shapes(input_data1, input_data2,
                                             param_name_input1="input_x",
                                             param_name_input2="input_y")
    dtype = input_x.dtype
    int_list = ("int8", "uint8", "int32")
    int_flag = dtype in int_list
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")
    data_x_broad = tbe.broadcast(input_x, shape_list[2])
    data_y_broad = tbe.broadcast(input_y, shape_list[2])
    res = tbe.vdiv(data_x_broad, data_y_broad)

    if int_flag:
        res = tbe.floor(res)

    res = tbe.cast_to(res, dtype)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def div(input_x, input_y, output_div, kernel_name="div"):
    """
    algorithm: div
    calculating data's div, res =x / y

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_div: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "div"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    dtype = input_x.get("dtype")
    shape_y = input_y.get("shape")

    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_y, param_name="input_y")
    shape_list = shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="input_x",
                                             param_name_input2="input_y")
    input_dtype = dtype.lower()
    check_list = ("float16", "float32", "int8", "uint8", "int32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    reshape_x, reshape_y = shape_util.refine_shapes_for_broadcast(shape_list[0],
                                                                  shape_list[1])
    data_x = tvm.placeholder(reshape_x, dtype=input_dtype, name="data_x")
    data_y = tvm.placeholder(reshape_y, dtype=input_dtype, name="data_y")

    res = div_compute(data_x, data_y, output_div, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    tbe.cce_build_code(sch, config)
