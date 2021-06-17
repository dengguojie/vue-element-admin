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

sigmoid_grad
"""
from functools import reduce as reduceIns

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator

# General limitation of the reduce size for input shape: 2**30
SHAPE_SIZE_LIMIT = 2 ** 30


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=invalid-name,too-many-locals,redefined-argument-from-local
def sigmoid_grad_compute(x, y, z, kernel_name="sigmoid_grad"):
    """
    algorithm : sigmoid grad compute

    sigmoid_grad = (sigmoid - sigmoid*sigmoid)*grad

    Parameters:
    ----------
    x : a tensor of input data

    y : a tensor of grad

    z : output dict

    kernel_name : cce kernel name, default value is "sigmoid_grad"

    Returns
    -------
    a tenosr
    """
    dtype = x.dtype.lower()
    cast_support = tbe_platform.api_check_support(
        "te.lang.cce.cast_to", "f322f16")
    if dtype == "float32" and not cast_support:
        error_detail = "float32 transfer to float16 is only supported on mini and cloud platform!"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x", 'y', error_detail)
    vmul_support = tbe_platform.api_check_support(
        "te.lang.cce.vmul", "float32")
    vsub_support = tbe_platform.api_check_support(
        "te.lang.cce.vsub", "float32")
    if dtype == "float16":
        x = tbe.cast_to(x, "float32")
        y = tbe.cast_to(y, "float32")
    sigmoid_square = tbe.vmul(x, x)
    if dtype == "float32" and not vmul_support:
        sigmoid_square = tbe.cast_to(sigmoid_square, "float16")
    tensor_sub = tbe.vsub(x, sigmoid_square)
    if dtype == "float32" and not vsub_support:
        tensor_sub = tbe.cast_to(tensor_sub, "float16")
    res = tbe.vmul(tensor_sub, y)
    if dtype == "float16":
        res = tbe.cast_to(res, "float16")
    return res


@register_operator("SigmoidGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def sigmoid_grad(x,
                 dx,
                 out,
                 kernel_name="sigmoid_grad"):
    """
    do sigmoid grad

    sigmoid_grad = (sigmoid - sigmoid*sigmoid)*grad

    Parameters:
    ----------
    x : dictionary shape of sigmoid input

    dx : dictionary shape of grad

    out: dictionary output

    kernel_name : cce kernel name, default value is "sigmoid_grad_cce"

    Returns
    -------
    None
    """
    x_dtype = x.get("dtype").lower()
    dx_dtype = dx.get("dtype").lower()
    shape_x = x.get("shape")
    shape_dx = dx.get("shape")
    check_list = ("float16", "float32")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")
    para_check.check_dtype(dx_dtype, check_list, param_name="input_dx")
    para_check.check_elewise_shape_range([x, dx], support_broadcast=False)
    if len(shape_x) != len(shape_dx):
        error_detail = "The shape of two input parameters are not match for dynamic sigmoid_grad."
        error_manager_vector.raise_err_two_input_dtype_invalid("sigmoid_grad", "x", "dx", error_detail)
    if x_dtype != dx_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "x", "dx",
                                                              x_dtype, dx_dtype)
    ins = classify([x, dx], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (sig, dx) in ins:
        with tbe.compute():
            shape_sig, shape_dx = shape_util.variable_shape([sig, dx])
            tensor_sig = tvm.placeholder(shape_sig, x_dtype, "tensor_x")
            tensor_dx = tvm.placeholder(shape_dx, dx_dtype, "tensor_dx")
            res = sigmoid_grad_compute(tensor_sig, tensor_dx, out, kernel_name)
            tensors.append([tensor_sig, tensor_dx, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
