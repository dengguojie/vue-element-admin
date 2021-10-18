"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

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

import te.lang.cce as tbe
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import para_check
from te.utils import shape_util


@fusion_manager.register("lerp")
# pylint: disable=unused-argument
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
    shape_x = shape_util.shape_to_list(start.shape)
    shape_y = shape_util.shape_to_list(end.shape)
    shape_z = shape_util.shape_to_list(weight.shape)

    shape_x, shape_y, shape_tmp = shape_util.produce_shapes(shape_x, shape_y)
    shape_tmp, shape_z, shape_max = shape_util.produce_shapes(shape_tmp, shape_z)
    para_check.check_shape(shape_max)

    start = tbe.broadcast(start, shape_max)
    end = tbe.broadcast(end, shape_max)
    weight = tbe.broadcast(weight, shape_max)
    tmp = tbe.vsub(end, start)
    res_tmp = tbe.vmul(weight, tmp)
    res = tbe.vadd(start, res_tmp)

    return res


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
    shape_x = start.get("shape")
    shape_y = end.get("shape")
    shape_z = weight.get("shape")

    dtype = start.get("dtype")

    input_dtype = dtype.lower()

    para_check.check_shape_rule(shape_x)
    para_check.check_shape_rule(shape_y)
    para_check.check_shape_rule(shape_z)

    para_check.check_shape(shape_x)
    para_check.check_shape(shape_y)
    para_check.check_shape(shape_x)

    para_check.check_kernel_name(kernel_name)

    check_tuple = ("float16", "float32")
    para_check.check_dtype_rule(input_dtype, check_tuple)

    shape_x, shape_y, shape_tmp = shape_util.produce_shapes(shape_x, shape_y)
    shape_tmp, shape_z, shape_max = shape_util.produce_shapes(shape_tmp, shape_z)

    para_check.check_shape(shape_max)

    data_x = tvm.placeholder(shape_x, name="data_1", dtype=input_dtype)
    data_y = tvm.placeholder(shape_y, name="data_2", dtype=input_dtype)
    data_z = tvm.placeholder(shape_z, name="data_3", dtype=input_dtype)

    res = lerp_compute(data_x, data_y, data_z, y, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, data_z, res]}

    tbe.build(schedule, config)
