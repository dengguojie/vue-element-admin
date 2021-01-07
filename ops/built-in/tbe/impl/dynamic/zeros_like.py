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

zeros_like
"""
from functools import reduce as functools_reduce

import te.lang.cce as tbe
import te.lang.base as tbe_base
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode


# pylint: disable=locally-disabled,invalid-name,unused-argument
def zeros_like_compute(x, y, kernel_name="zeros_like"):
    """
    Enter a tensor, output a tensor of all zero,
    you can specify the output data type

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input data
    y: TVM tensor
        the placeholder of output data
    kernel_name : str
        cce kernel name, default value is "zeros_like"

    Returns
    -------
    res: TVM tensor
        the result of zeros_like_compute
    """
    src_dtype = x.dtype.lower()
    dst_type = src_dtype
    src_type_list = ("int8", "uint8")
    dst_type_list = ("int8", "uint8")
    if src_dtype in src_type_list:
        src_dtype = "float16"
    zero = tvm.const(0, dtype=src_dtype)
    zero_src = tbe.broadcast(zero, x.shape)
    if src_dtype in dst_type_list:
        zero_src = tbe.cast_to(zero_src, dst_type,
                                           f1628IntegerFlag=True)
    else:
        zero_src = tbe.cast_to(zero_src, dst_type)
    return zero_src


@tbe_base.register_operator("ZerosLike")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def zeros_like(x, y, kernel_name="zeros_like"):
    """
    output a tensor of all zero, you can specify the output type

    Parameters
    ----------
    x: dict
        shape and dtype of input, only support float16, float32,
        int32,int8,uint8,bool
    y: dict
        shape and dtype of output data
    kernel_name: str
        cce kernel name, default value is "zeros_like"

    Returns
    ------
    None
    """
    dtype_x = x.get("dtype")
    check_list_src = ("float16", "float32", "int32", "int8", "uint8", "bool")
    src_dtype = dtype_x.lower()
    para_check.check_dtype(src_dtype, check_list_src, param_name="x")
    schedules, tensors = [], []
    ins = classify([x], Mode.ELEWISE)
    for (input_x,) in ins:
        with tbe_base.compute():
            shape_x = shape_util.variable_shape([input_x])
            shape_x = (functools_reduce(lambda x, y: x * y, shape_x[0]),)
            x_input = tvm.placeholder(shape_x, name="x_input", dtype=src_dtype)
            res = zeros_like_compute(x_input, y, kernel_name=kernel_name)
            tensors.append([x_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
