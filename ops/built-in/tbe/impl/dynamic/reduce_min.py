"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

dynamic reduce_min
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# pylint: disable=unused-argument,invalid-name
@register_operator_compute("ReduceMin", op_mode="dynamic", support_fusion=True)
def reduce_min_compute(x, axes, y, keepdims=None,
                       kernel_name="reduce_min"):
    """
    reduce_min compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axes: int, list, tuple or NONETYPE
        the axes for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_min".

    Returns
    -------
    res: TVM tensor
         output tensor, has the same shape and type as input tensor.
    """
    dtype = x.dtype
    if dtype in ("int8", "uint8"):
        x = tbe.cast_to(x, "float16")
    elif not tbe_platform.api_check_support("tbe.dsl.reduce_min", dtype):
        x = tbe.cast_to(x, "float32")
    res_min = tbe.reduce_min(x, axis=axes, keepdims=keepdims)
    res = tbe.cast_to(res_min, dtype)

    return res


# pylint: disable=too-many-locals,invalid-name
@register_operator("ReduceMin")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_min(x, axes, y, keepdims=False, kernel_name="reduce_min"):
    """
    reduce a tensor on a certain axes based on min.

    Parameters
    ----------
    x : dict
        shape and dtype of input
    axes : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input
    keepdims: bool
        if true, retains reduced dimensions with length 1,
        default value is None
    kernel_name : str
        kernel name, default value is "reduce_min"

    Returns
    -------
    None
    """

    dtype_x = x["dtype"]
    dtype_lower_x = dtype_x.lower()
    check_list_x = ("float16", "float32", "int8", "uint8", "int32")
    para_check.check_dtype(dtype_lower_x, check_list_x)
    x["rel_pos_to_reduce"] = "before"

    dtype_axes = axes["dtype"]
    dtype_lower_axes = dtype_axes.lower()
    check_list_axes = ("int32", "int64")
    para_check.check_dtype(dtype_lower_axes, check_list_axes, param_name="axes")
    axes["rel_pos_to_reduce"] = "axis"

    schedules = []
    tensors = []
    ins = classify([x, axes], OpPatternMode.REDUCE, {"keepdims": keepdims is True})

    for (_x, _axes) in ins:
        with tbe.compute():
            shape_x, shape_axes = shape_util.variable_shape([_x, _axes], op_mode="reduce")
            data_input_x = tvm.placeholder(shape_x, name="data_input_x",
                                           dtype=dtype_lower_x)
            data_input_axes = tvm.placeholder(shape_axes, name="data_input_axes",
                                              dtype=dtype_lower_axes)
            axes_d = shape_util.axis_check(len(shape_x), _axes.get("value"))
            res = reduce_min_compute(data_input_x, axes_d, y, keepdims)
            tensors.append([data_input_x, data_input_axes, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
