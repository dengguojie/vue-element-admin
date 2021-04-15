#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

strided_slice_grad
"""
from impl.dynamic.pad import PadInit
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector


def _check_mask(input_mask, is_shrink=False):
    """ Check whether the value of the input mask is 0.

    Parameters
    ----------
    input_mask: int.
        value of the input mask.

    Returns
    -------
    None.
    """
    if is_shrink:
        if input_mask not in (0, 2):
            error_manager_vector.raise_err_input_value_invalid("strided_slice_grad", "shrink_axis_mask",
                                                               "(0, 2)", str(input_mask))
    elif input_mask != 0:
        error_manager_vector.raise_err_input_value_invalid("strided_slice_grad", "new_axis_mask",
                                                               "0", str(input_mask))


# pylint: disable=locally-disabled,too-many-arguments,invalid-name
@register_operator("StridedSliceGrad")
def strided_slice_grad(shape, begin, end, strides, dy, output, begin_mask=0,
                       end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0,
                       kernel_name="strided_slice_grad"):
    """ Since `StridedSlice` cuts out pieces of its `input` which is size`shape_dy`, its gradient
    will have the same shape (which is passed here as `shape_x`). The gradient will be zero in any
    element that the slice does not select.

    Parameters
    ----------
    shape : list or tuple.
        shape of input
    begin: list or tuple.
        represents the index of the first value to select.
    end: list or tuple.
        represents the index of the last value to select.
    strides: list or tuple.
        step length to select.
    dy : dict
        shape and dtype of input
    output : dict
        shape and dtype of out
    begin_mask: int
        a bitmask where a bit i being 1 means to ignore the begin value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th position is actually an ellipsis.
    new_axis_mask: int
        a bitmask where bit `i` being 1 means the `i`th specification creates a
        new shape 1 dimension.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th specification should shrink
        the dimensionality.
    kernel_name : str
        cce kernel name, default value is "strided_slice_grad"

    Returns
    -------
    None.
    """
    dtype = dy.get("dtype").lower()
    para_check.check_dtype(dtype, ("float16", "float32", "int32"), param_name="dy")
    _check_mask(new_axis_mask)
    _check_mask(shrink_axis_mask, True)

    obj = PadInit(kernel_name)
    obj.init_src_dst_gm((shape, begin, end, strides, dy), (output,), pad_input_idx=4, pad_outnput_idx=0)

    outer_compile = dict()
    outer_compile["begin_mask"] = begin_mask
    outer_compile["end_mask"] = end_mask
    outer_compile["ellipsis_mask"] = ellipsis_mask
    outer_compile["new_axis_mask"] = new_axis_mask
    outer_compile["shrink_axis_mask"] = shrink_axis_mask
    return obj.pad_compute(outer_compile)
