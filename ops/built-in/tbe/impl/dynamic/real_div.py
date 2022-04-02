#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

real_div
"""
from __future__ import absolute_import

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import tbe_platform


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=too-many-locals,redefined-argument-from-local
def real_div_compute(x1, x2, y, kernel_name="real_div", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    calculating data's realdiv, c = a / b

    Parameters
    ----------
    x1: TVM tensor
        the placeholder of first input data
    x2: TVM tensor
        the placeholder of second input data
    y: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is real_div

    Returns
    -------
    res : output of the data's divide
    """
    dtype = x1.dtype
    has_improve_precision = False
    if dtype == "float16" and impl_mode != OpImplMode.HIGH_PERFORMANCE and \
            tbe_platform.api_check_support("te.lang.cce.vdiv", "float32"):
        x1 = tbe.cast_to(x1, "float32")
        x2 = tbe.cast_to(x2, "float32")
        has_improve_precision = True

    shape_x = shape_util.shape_to_list(x1.shape)
    shape_y = shape_util.shape_to_list(x2.shape)
    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y)
    data_x = tbe.broadcast(x1, shape_max)
    data_y = tbe.broadcast(x2, shape_max)
    res = tbe.vdiv(data_x, data_y)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("RealDiv")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def real_div(x1, x2, y, kernel_name="real_div", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    algorithm: real_div
    calculating data's real_div, c = a / b

    Parameters
    ----------
    x1 : dict
        shape and dtype of first input, only support float16, float32, int32
    x2 : dict
        shape and dtype of second input, only support float16, float32, int32
    y: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name : str
        cce kernel name, default value is real_div

    Returns
    -------
    None
    """

    x_dtype = x1.get("dtype").lower()
    y_dtype = x2.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")
    para_check.check_dtype(y_dtype, check_list, param_name="input_y")
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "x1", "x2",
                                                              x_dtype, y_dtype)
    ins = classify([x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (x1, x2) in ins:
        with tbe.compute():
            x_shape, y_shape = shape_util.variable_shape([x1, x2])
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            tensor_y = tvm.placeholder(y_shape, y_dtype, "tensor_y")
            res = real_div_compute(tensor_x, tensor_y, y, kernel_name, impl_mode)

            tensors.append([tensor_x, tensor_y, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
