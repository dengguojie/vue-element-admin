"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

log_space_d
"""

import math
import te.lang.cce as tbe
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import para_check


@fusion_manager.register("log_space_d")
# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
def log_space_d_compute(assist, y, start, end, steps=100, base=10.0, dtype=1, kernel_name="log_space_d"):
    """
    calculating data

    Parameters
    ----------
    assist : TVM tensor
        the placeholder of assist
    y : dict
        dict of y, include keys(shape and dtype)
    start : float
        Set the starting value
    end : float
        Set the termination value
    steps : int
        The number of sample points
    base : float
        The base of the exponential function
    dtype : int
        The dtype of output
    kernel_name : str
        kernel name, default value is "log_space_d"

    Returns
    -------
    output tensor
    """
    output_dtype_dict = {0:"float16", 1:"float32"}
    step_minus_one = tvm.const(steps - 1, "int32")
    scalar_one = tvm.const(1.0, "float32")
    if steps <= 1:
        diff = end - start
    else:
        diff = (end - start) / (step_minus_one)
    if assist.dtype != "float32":
        assist = tbe.cast_to(assist, "float32")
    diff = tbe.vmuls(assist, diff)
    x = tbe.vadds(diff, start)
    if base > 0:
        log_base = math.log(base)
        log_base = tvm.const(log_base, "float32")
        index = tbe.vmuls(x, log_base)
        rs = tbe.vexp(index)
    elif base < 0:
        scalar_two = tvm.const(2.0, "float32")
        one_half = 1.0 / scalar_two
        negative_two = tvm.const(-2.0, "float32")
        x_abs = tbe.vabs(x)
        x_div_two = tbe.vmuls(x_abs, one_half)
        x_div_two_floor = tbe.floor(x_div_two)
        x_remainder1 = tbe.vmuls(x_div_two_floor, scalar_two)
        x_remainder2 = tbe.vsub(x_abs, x_remainder1)
        negative_two_x = tbe.vmuls(x_remainder2, negative_two)
        rs1 = tbe.vadds(negative_two_x, scalar_one)
        abs_base = math.fabs(base)
        log_base = math.log(abs_base)
        log_base = tvm.const(log_base, "float32")
        index = tbe.vmuls(x, log_base)
        rs2 = tbe.vexp(index)
        rs = tbe.vmul(rs1, rs2)
    elif base == 0:
        scalar_zero = tvm.const(0.0, "float32")
        rs = tbe.vcmpsel(x, scalar_zero, 'eq', scalar_one, scalar_zero)
    if steps == 0:
        rs = assist
    if output_dtype_dict[dtype] == "float16":
        rs = tbe.cast_to(rs, output_dtype_dict[dtype])
    return  rs


# 'pylint: disable=too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def log_space_d(assist, y, start, end, steps=100, base=10.0, dtype=1, kernel_name="log_space_d"):
    """
    calculating data

    Parameters
    ----------
    assist : dict
        shape and dtype of input
    y : dict
        shape and dtype of output
    start : float
        Set the starting value
    end : float
        Set the termination value
    steps : int
        The number of sample points
    base : float
        The base of the exponential function
    dtype : int
        The dtype of output
    kernel_name : str
        kernel name, default value is "log_space_d"

    Returns
    -------
    None
    """

    shape = assist.get("shape")

    assist_dtype = assist.get("dtype").lower()

    para_check.check_shape_rule(shape)
    para_check.check_shape(shape)
    para_check.check_kernel_name(kernel_name)

    check_tuple = ("float16", "float32")
    para_check.check_dtype_rule(assist_dtype, check_tuple)

    is_onedim = len(shape)

    if steps < 0:
        raise RuntimeError("please input steps > 0")
    if is_onedim != 1:
        raise RuntimeError("assist.shape only support one dim")
    if shape[0] != steps:
        raise RuntimeError("assist shape should equal to steps")
    if dtype not in [0, 1]:
        raise RuntimeError("only support dtype 0,1")

    data_assist = tvm.placeholder(shape, name="data_assist", dtype=assist_dtype)
    res = log_space_d_compute(data_assist, y, start, end, steps, base, dtype, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_assist, res]}

    tbe.build(schedule, config)
