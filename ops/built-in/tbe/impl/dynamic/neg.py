#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

neg
"""
from __future__ import absolute_import
from functools import reduce as reduceIns

import te.lang.cce as tbe
from te import tvm
import te.lang.base as tbe_base
from te.utils import shape_util
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils import para_check
from impl.util.platform_adapter import register_operator

# define a scaler, value = -1
SCALER_NEGATIVE_ONE = -1
# define a float scaler, value = -1.0
SCALER_NEGATIVE_ONE_FLOAT = -1.0


# pylint: disable=locally-disabled,unused-argument,redefined-argument-from-local
def neg_compute(input_x, output_y, kernel_name="neg"):
    """
    compute neg

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        data of output.
    kernel_name: str
        kernel name, default value is "neg"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype = input_x.dtype.lower()
    shape = shape_util.shape_to_list(input_x.shape)

    if dtype == "int32":
        data_tmp = tbe.broadcast(SCALER_NEGATIVE_ONE, shape)
        res = tbe.vmul(input_x, data_tmp)
    else:
        if dtype == "int8":
            input_x = tbe.cast_to(input_x, "float16")
        res = tbe.vmuls(input_x, SCALER_NEGATIVE_ONE_FLOAT)

    if dtype == "int8":
        res = tbe.cast_to(res, dtype)

    return res


@register_operator("Neg")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def neg(input_x, output_y, kernel_name="neg"):
    """
    Computes numerical negative value element-wise, y = -x.

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32, int32, int8
    output_y: dict
        shape and dtype of output, should be same type as input
    kernel_name: str
        kernel name, default value is "neg"

    Returns
    -------
    None
    """
    dtype_input = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "int8")
    para_check.check_dtype(dtype_input, check_list, param_name="input_x")

    ins = classify([input_x], Mode.ELEWISE)
    schedules, tensors = [], []
    for (input_x,) in ins:
        with tbe_base.compute():
            x_shape = shape_util.variable_shape([input_x])

            fuse_shape = [1]
            fuse_shape[0] = reduceIns(lambda x, y: x * y, x_shape[0])
            data_input = tvm.placeholder(fuse_shape, name="data_input",
                                         dtype=dtype_input)
            res = neg_compute(data_input, output_y, kernel_name)

            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
