#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

floor
"""
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as reduceIns
from te.utils.op_utils import *

# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("floor")
def floor_compute(input_x, output_y, kernel_name="floor"):
    """
    floor compute
    calculating element-wise largest integer not greater than input_x

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    output_y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "floor"

    Returns
    -------
    res: TVM tensor
        the result of floor(input_x)
    """
    res_int32 = te.lang.cce.floor(input_x)
    res = te.lang.cce.cast_to(res_int32, input_x.dtype)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def floor(input_x, output_y, kernel_name="floor"):
    """
    algorithm: floor
    calculating element-wise largest integer not greater than input_x,
    the type of input_x is float16 or float32

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input
    output_y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "floor"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()

    check_shape(shape, param_name="input_x")
    check_list = {"float16", "float32"}
    check_dtype(dtype, check_list, param_name="input_x")

    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape)
    data = tvm.placeholder(fuseshape, dtype=dtype, name="data")
    res = floor_compute(data, output_y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data, res]}
    te.lang.cce.cce_build_code(sch, config)
