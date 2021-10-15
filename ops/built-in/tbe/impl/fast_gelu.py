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

fast_gelu
"""
import functools

import te.lang.cce as tbe
from te import tvm
from te import platform as tbe_platform
from te.utils import para_check


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
# pylint: disable=too-many-locals,unused-variable
@tbe_platform.fusion_manager.fusion_manager.register("fast_gelu")
def fast_gelu_compute(input_x, output_y, kernel_name="fast_gelu",
                      impl_mode="high_performance"):
    """
    mathematical formula of fast_gelu(x):
    fast_gelu(x) = xe^(0.851x)(x-|x|)/(1+e^(-1.702|x|))
    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input input_x
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is fast_gelu

    Returns
    -------
     A TVM tensor same as input placeholders.
    """
    attr = 1.702
    dtype = input_x.dtype.lower()
    attr_opp = 0 - attr
    attr_half = attr / 2
    const_0 = tvm.const(attr_opp, dtype)
    const_1 = tvm.const(1, dtype)
    abs_x = tbe.vabs(input_x)
    mul_abs_x = tbe.vmuls(abs_x, const_0)
    exp_abs_x = tbe.vexp(mul_abs_x)
    div_down = tbe.vadds(exp_abs_x, const_1)

    const_2 = tvm.const(attr_half, dtype)
    pn_x = tbe.vsub(input_x, abs_x)
    mul_pn_x = tbe.vmuls(pn_x, const_2)
    exp_pn_x = tbe.vexp(mul_pn_x)
    div_up = tbe.vmul(input_x, exp_pn_x)

    if impl_mode == "high_performance":
        div_down_rec = tbe.vrec(div_down, priority_flag=0)
    else:
        div_down_rec = tbe.vrec(div_down, priority_flag=1)
    result = tbe.vmul(div_up, div_down_rec)

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def fast_gelu(input_x, output_y, kernel_name="fast_gelu",
              impl_mode="high_performance"):
    """
    mathematical formula of fast_gelu(x):
    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is fast_fast_gelu

    Returns
    -------
    none.
    """
    attr = 1.702
    shape = input_x.get("shape")
    para_check.check_shape(shape, param_name="input_x")

    check_list = ("float16", "float32")
    input_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, shape)
    data = tvm.placeholder(fuseshape, name="data", dtype=input_dtype)
    result = fast_gelu_compute(data, output_y, kernel_name, impl_mode)

    with tvm.target.cce():
        sch = tbe.auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, result]}

    tbe.cce_build_code(sch, config)
