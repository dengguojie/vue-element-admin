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

dynamic reduce_max_d
"""
import te.lang.cce as tbe
import te.lang.base as tbe_base
from te.utils import para_check
from te.utils import shape_util
from te import tvm
from te.lang.base.operation import add_compile_info

NONETYPE = type(None)


# 'pylint: disable=unused-argument,invalid-name
def reduce_max_d_compute(x, y, axes=None, keepdims=None,
                         kernel_name="reduce_max_d"):
    """
    reduce_max_d compute

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
        cce kernel name, default value is "reduce_max_d".

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
@tbe_base.register_operator("ReduceMaxD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_max_d(x, y, axes=None, keepdims=None, kernel_name="reduce_max_d"):
    """
    reduce a tensor on a certain axes based on max.

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input
    axes: list
        the first axes to reduce,may be negative to index from the end
        (e.g., -1 for the last axes).
        axes may be int or list(e.g. [1,2])
    keepdims: bool
        if true, retains reduced dimensions with length 1,
        default value is None
    kernel_name : str
        kernel name, default value is "reduce_max_d"

    Returns
    -------
    None
    """

    dtype = x["dtype"]
    dtype_lower = dtype.lower()
    check_list = ("float16", "float32", "int8", "uint8", "int32")
    para_check.check_dtype(dtype_lower, check_list)
    add_compile_info("_ori_axis", axes)

    shape = x["shape"]
    shape_len = len(shape)
    if not axes:
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)
    axes = shape_util.axis_check(shape_len, axes)

    schedules = []
    tensors = []
    ins = tbe_base.shape_classifier.classify([x, axes], tbe_base.shape_classifier.Mode.REDUCE)

    for (x, axes) in ins:
        with tbe_base.compute():
            shape_var_new = shape_util.variable_shape([x])[0]
            data_input = tvm.placeholder(shape_var_new, name="data_input",
                                         dtype=dtype_lower)
            res = reduce_max_d_compute(data_input, y, axes, keepdims)
            tensors.append([data_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
