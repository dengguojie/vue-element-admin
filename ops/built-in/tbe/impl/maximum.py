#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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

maximum
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from topi import generic
from topi.cce import util
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import *

SHAPE_SIZE_LIMIT = 2147483648  # shape limit

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=invalid-name,unused-variable,too-many-locals
@fusion_manager.register("maximum")
def maximum_compute(x1, x2, y, kernel_name="maximum"):
    """maximum compute

    Parameters:
    ----------
    x1: TVM tensor
        input_x tensor.
    x2: TVM tensor
        input_y tensor.
    y: dict
        shape and dtype of output.
    kernel_name: str
        cce kernel name, default value is "maximum".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """

    shape_x = te.lang.cce.util.shape_to_list(x1.shape)
    shape_y = te.lang.cce.util.shape_to_list(x2.shape)
    shape1, shape2, shape_max = broadcast_shapes(shape_x, shape_y, param_name_input1="x1", param_name_input2="x2")

    data1 = te.lang.cce.broadcast(x1, shape_max)
    data2 = te.lang.cce.broadcast(x2, shape_max)
    res = te.lang.cce.vmax(data1, data2)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def maximum(x1, x2, y, kernel_name="maximum"):
    """
    do element-wise maximum operation between two input tensors

    Parameters:
    ----------
    x1 : dict
        shape and dtype of first input, only support float16, float32, int32
    x2 : dict
        shape and dtype of second input, only support float16, float32, int32
    y: dict
        shape and dtype of output, should be the broadcast shape and
        type as input
    kernel_name : str
        cce kernel name, default value is maximum

    Returns
    -------
    None
    """
    shape1 = util.scalar2tensor_one(x1.get("shape"))
    shape2 = util.scalar2tensor_one(x2.get("shape"))

    check_shape(shape1, param_name="x1")
    check_shape(shape2, param_name="x2")

    check_list = ["float16", "float32", "int32"]
    dtype = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()
    check_dtype(dtype, check_list, param_name="x1")
    check_dtype(dtype_x2, check_list, param_name="x2")

    shape_x, shape_y, _ = broadcast_shapes(shape1, shape2, param_name_input1="x1", param_name_input2="x2")

    data1 = tvm.placeholder(shape_x, dtype=dtype, name="data1")
    data2 = tvm.placeholder(shape_y, dtype=dtype, name="data2")

    res = maximum_compute(data1, data2, y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data1, data2, res]}
    te.lang.cce.cce_build_code(sch, config)
