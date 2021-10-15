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

fast_gelu grad
"""

import operator
import functools

import te.lang.cce as tbe
from te import tvm
from te import platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from tbe.dsl import broadcast
from impl.util.platform_adapter import error_manager_vector
from impl.util import util_compute


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
# pylint: disable=too-many-locals
@tbe_platform.fusion_manager.fusion_manager.register("fast_gelu_grad")
def fast_gelu_grad_compute(input_dy, input_x, output_z,
                           kernel_name="fast_gelu_grad",
                           impl_mode="high_performance"):
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
    const_3 = tvm.const(1, dtype)

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
    if util_compute.check_batchmatmul_fuse(input_dy):
        batch_shape = shape_util.shape_to_list(input_dy.op.attrs["batch_shape"])
        unfold_shape = batch_shape + shape_util.shape_to_list(input_dy.shape)[-4:]
        result_temp = broadcast(result_temp, unfold_shape)
        result_temp = util_compute.batchmatmul_elem_reshape(input_dy, result_temp, batch_shape, "fast_gelu_grad")
    elif util_compute.check_batchmatmul_fuse(input_x):
        batch_shape = shape_util.shape_to_list(input_x.op.attrs["batch_shape"])
        unfold_shape = batch_shape + shape_util.shape_to_list(input_x.shape)[-4:]
        input_dy = broadcast(input_dy, unfold_shape)
        input_dy = util_compute.batchmatmul_elem_reshape(result_temp, input_dy, batch_shape, "fast_gelu_grad")

    result = tbe.vmul(input_dy, result_temp)
    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME,
                            para_check.OPTION_ATTR_STR)
def fast_gelu_grad(input_dy, input_x, output_z, kernel_name="fast_gelu_grad",
                   impl_mode="high_performance"):
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
        error_manager_vector.raise_err_inputs_shape_not_equal("fast_gelu_grad", "shape_dy", "shape_x",
                                                              shape_dy, shape_x, shape_x)

    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, shape_dy)
    data_dy = tvm.placeholder(fuseshape, name="data_dy", dtype=input_dtype)
    data_x = tvm.placeholder(fuseshape, name="data_x", dtype=input_dtype)
    res = fast_gelu_grad_compute(data_dy, data_x, output_z, kernel_name, impl_mode)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_dy, data_x, res]}

    tbe.cce_build_code(sch, config)
