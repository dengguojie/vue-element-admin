"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

lerp
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator


@register_operator_compute("Lerp", op_mode="dynamic", support_fusion=True)
# 'pylint: disable=unused-argument
def lerp_compute(start, end, weight, y, kernel_name="lerp"):
    """
    Compute

    Parameters
    ----------
    start: dict
        data of input
        datatype suports float32,float16
    end: dict
        data of input
        datatype suports float32,float16
    weight: dict
        data of input
        datatype suports float32,float16
    y: dict
        data of output
    kernel_name: str
        the name of the operator
    Returns
    -------
    None
    """

    # Broadcast the shape of start, end and weight
    shape_x = shape_util.shape_to_list(start.shape)
    shape_y = shape_util.shape_to_list(end.shape)
    shape_z = shape_util.shape_to_list(weight.shape)
    shape_x, shape_y, shape_tmp = shape_util.broadcast_shapes(shape_x, shape_y)
    shape_tmp, shape_z, shape_max = shape_util.broadcast_shapes(shape_tmp, shape_z)
    start = tbe.broadcast(start, shape_max)
    end = tbe.broadcast(end, shape_max)
    weight = tbe.broadcast(weight, shape_max)

    # Computational logic：out = start+(end-start)*weight
    sub_val = tbe.vsub(end, start)
    mul_val = tbe.vmul(weight, sub_val)
    res = tbe.vadd(start, mul_val)

    return res


# 'pylint: disable=too-many-locals
@register_operator("Lerp")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def lerp(start, end, weight, y, kernel_name="lerp"):
    """
    Lerp

    Parameters
    ----------
    start: dict
        data of input
        datatype suports float32,float16
    end: dict
        data of input
        datatype suports float32,float16
    weight: dict
        data of input
        datatype suports float32,float16
    y: dict
        data of output
    kernel_name: str
        the name of the operator
    Returns
    -------
    None
    """
    dtype = start.get("dtype")
    input_dtype = dtype.lower()
    para_check.check_kernel_name(kernel_name)
    check_tuple = ("float16", "float32")
    para_check.check_dtype_rule(input_dtype, check_tuple)
    ins = classify([start, end, weight], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for(_x, _y, _z) in ins:
        with tbe.compute():
            shape_x, shape_y, shape_z = shape_util.variable_shape([_x, _y, _z])
            data_x = tvm.placeholder(shape_x, name="data_1", dtype=input_dtype)
            data_y = tvm.placeholder(shape_y, name="data_2", dtype=input_dtype)
            data_z = tvm.placeholder(shape_z, name="data_3", dtype=input_dtype)
            res = lerp_compute(data_x, data_y, data_z, y, kernel_name)
            tensors.append([data_x, data_y, data_z, res])
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)
    config = {"name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
