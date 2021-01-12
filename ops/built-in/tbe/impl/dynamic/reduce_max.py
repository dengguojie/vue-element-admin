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

dynamic reduce_max
"""
import te
import te.lang.cce as tbe
import te.lang.base as tbe_base
from te import tvm
from te import platform as tbe_platform
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils import shape_util
from te.utils import para_check
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import OPTION_ATTR_BOOL
from te.utils.op_utils import KERNEL_NAME

NONETYPE = type(None)


# pylint: disable=unused-argument,invalid-name
# pylint: disable=redefined-argument-from-local
def reduce_max_compute(x, axes, y, keepdims=None,
                         kernel_name="reduce_max"):
    """
    reduce_max compute

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
        cce kernel name, default value is "reduce_max".

    Returns
    -------
    res: TVM tensor
         output tensor, has the same shape and type as input tensor.
    """
    dtype = x.dtype
    if dtype in ("int8", "uint8"):
        x = tbe.cast_to(x, "float16")
    res_max = tbe.reduce_max(x, axis=axes, keepdims=keepdims)
    res = tbe.cast_to(res_max, dtype)

    return res


# 'pylint: disable=too-many-locals,invalid-name
@te.op.register_operator("ReduceMax")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def reduce_max(x, axes, y, keepdims=False, kernel_name="reduce_max"):
    """
    reduce a tensor on a certain axes based on max.

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
        kernel name, default value is "reduce_max"

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
    check_dtype(dtype_lower_axes, check_list_axes, param_name="axes")
    axes["rel_pos_to_reduce"] = "axis"

    schedules = []
    tensors = []
    ins = classify([x, axes], Mode.REDUCE)

    for (x, axes) in ins:
        with tbe_base.compute():
            shape_x, shape_axes = shape_util.variable_shape([x, axes], op_mode="reduce")
            data_input_x = tvm.placeholder(shape_x, name="data_input_x",
                                           dtype=dtype_lower_x)
            data_input_axes = tvm.placeholder(shape_axes, name="data_input_axes",
                                              dtype=dtype_lower_axes)
            axes_d = shape_util.axis_check(len(shape_x), axes.get("value"))
            res = reduce_max_compute(data_input_x, axes_d, y, keepdims)
            tensors.append([data_input_x, data_input_axes, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
