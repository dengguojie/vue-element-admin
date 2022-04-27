"""
Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

fill
"""
from functools import reduce
from operator import mul
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.util_common import gen_range


# 'pylint: disable=unused-argument,invalid-name
def check_supported(dims, value, y, kernel_name="fill"):
    """
    verify the types of fill supported by tbe
    """
    return True, ""


# 'pylint: disable=unused-argument,invalid-name,too-many-locals
def fill_compute(dims, value, y, kernel_name="fill"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    value : a number of float or int
    dtype : string
        the type of input
    kernel_name : str
        kernel name, default value is "fills"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """

    res = tbe.broadcast(value, dims)

    return res


@register_operator("Fill")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def fill(dims, value, y, kernel_name="fill"):
    """
    do  fill operation

    Parameters:
    ----------
    dims : the dict of input
    value :  the dict of input
    y:  the dict of output
    kernel_name : cce kernel name, default value is "fill"

    Returns
    -------
    None
    """
    # get the shape and dtype
    tmp_dtype = value.get("dtype").lower()
    dtype = tmp_dtype if tmp_dtype != "bool" else "int8"
    dtype_dims = dims.get("dtype").lower()
    dims["shape"] = [-1]
    dims['range'] = [[1, None]]

    const_value = dims.get('const_value')
    if const_value:
        const_value = list(const_value)
        shape_shape_adapt = [reduce(mul, const_value)]
        shape_range_adapt = gen_range(const_value)
    else:
        shape_shape_adapt = [-1]
        shape_range_adapt = [[1, None]]

    dims["shape"] = shape_shape_adapt
    dims['range'] = shape_range_adapt

    # check whether dtypes are right
    check_list = ("int8", "int32", "float16", "float32")
    para_check.check_dtype(dtype, check_list)

    extra_params = {"disable_optimization": True}
    ins = classify([dims, value], OpPatternMode.ELEWISE_WITH_BROADCAST, extra_params)
    schedules, tensors = [], []
    for (_dims, _value) in ins:
        with tbe.compute():
            shape_dim, shape = shape_util.variable_shape([_dims, _value])
            x_input = tvm.placeholder(shape, name="x_input", dtype=dtype)
            dim_input = tvm.placeholder(shape_dim, name="dim_input", dtype=dtype_dims)

            res = fill_compute(shape_dim, x_input, y, kernel_name=kernel_name)
            tensors.append([dim_input, x_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
