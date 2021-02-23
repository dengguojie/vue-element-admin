#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

fast_gelu grad
"""

import operator
import functools

import te.lang.cce as tbe
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector

import te.lang.base as tbe_base
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator


CONST_1 = 1


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
# pylint: disable=too-many-locals
@register_operator_compute("FastGeluGrad", op_mode="dynamic", support_fusion=False)
def fast_gelu_grad_compute(input_dy, input_x, output_z, kernel_name="fast_gelu_grad", impl_mode="high_performance"):
    """
    algorithm: fast_gelu_grad
    calculating: dy*res'
    res' = div_up/div_down
    div_up = e^(-1.702x) + 1.702xe^(-1.702x) + e^(1.702(x-|x|))
    div_down = (e^(-1.702x)+1)^2

    Parameters
    ----------
    input_dy : dict
        shape and dtype of dy input, only support float16, float32
    input_x : dict
        shape and dtype of x input, only support float16, float32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is fast_gelu_grad

    Returns
    -------
    A TVM tensor same as input placeholders.
    """
    attr = 1.702
    dtype = input_x.dtype
    attr_opp = 0 - attr
    const_1 = tvm.const(attr_opp, dtype)
    const_2 = tvm.const(attr, dtype)
    const_3 = tvm.const(CONST_1, dtype)

    # e^(-1.702x)
    abs_x = tbe.vabs(input_x)
    mul_abs_x = tbe.vmuls(abs_x, const_1)
    exp_x = tbe.vexp(mul_abs_x)

    # 1.702xe^(-1.702x)
    add_2 = tbe.vmul(input_x, exp_x)
    add_2 = tbe.vmuls(add_2, const_2)

    # e^(1.702(x-|x|))
    pn_x = tbe.vsub(input_x, abs_x)
    mul_pn_x = tbe.vmuls(pn_x, const_2)
    exp_pn_x = tbe.vexp(mul_pn_x)

    #  e^(-1.702x) + 1.702xe^(-1.702x) + e^(1.702(x-|x|))
    div_up = tbe.vadd(exp_x, add_2)
    div_up = tbe.vadd(div_up, exp_pn_x)

    # (e^(-1.702x)+1)^2
    div_down_i = tbe.vadds(exp_x, const_3)
    div_down = tbe.vmul(div_down_i, div_down_i)

    if impl_mode == "high_performance":
        div_down_rec = tbe.vrec(div_down, priority_flag=0)
    else:
        div_down_rec = tbe.vrec(div_down, priority_flag=1)
    result_temp = tbe.vmul(div_up, div_down_rec)

    result = tbe.vmul(input_dy, result_temp)
    return result


@register_operator("FastGeluGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def fast_gelu_grad(input_dy, input_x, output_z, kernel_name="fast_gelu_grad", impl_mode="high_performance"):
    """
    algorithm: fast_gelu_grad
    calculating: dy*res'
    res' = div_up/div_down
    div_up = e^(-1.702x) + 1.702xe^(-1.702x) + e^(1.702(x-|x|))
    div_down = (e^(-1.702x)+1)^2

    Parameters
    ----------
    input_dy : dict
        shape and dtype of dy input, only support float16, float32
    input_x : dict
        shape and dtype of x input, only support float16, float32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is fast_gelu_grad

    Returns
    -------
    none.
    """
    shape_dy = input_dy.get("shape")
    shape_x = input_x.get("shape")

    para_check.check_shape(shape_dy, param_name="input_dy")
    para_check.check_shape(shape_x, param_name="input_x")
    input_dtype = input_dy.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_dy")
    shape_dy = list(shape_dy)
    shape_x = list(shape_x)
    if not operator.eq(shape_dy, shape_x):
        error_detail = "all input shape must be equal"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "shape_dy", "shape_x", error_detail)

    ins = classify([input_dy, input_x], Mode.ELEWISE)
    schedules, tensors = [], []
    for (_input_dy, _input_x) in ins:
        with tbe_base.compute():
            dy_shape, x_shape = shape_util.variable_shape([_input_dy, _input_x])
            fuseshape = [1]
            fuseshape[0] = functools.reduce(lambda x, y: x * y, dy_shape)

            tensor_dy = tvm.placeholder(dy_shape, input_dtype, "tensor_dy")
            tensor_x = tvm.placeholder(x_shape, input_dtype, "tensor_x")
            res = fast_gelu_grad_compute(tensor_dy, tensor_x, output_z, kernel_name, impl_mode)
            tensors.append([tensor_dy, tensor_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
