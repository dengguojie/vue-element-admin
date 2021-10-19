#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

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
# pylint: disable=invalid-name, too-many-instance-attributes
# pylint: disable=too-many-arguments, useless-object-inheritance
# pylint: disable=too-many-locals, too-many-statements
# pylint: disable=attribute-defined-outside-init, unused-argument
# pylint: disable=attribute-defined-outside-init, chained-comparison

# pylint: disable=unused-argument
# pylint: disable=consider-using-in,unnecessary-pass
def check_supported(shape, begin, end, strides, dy, output, begin_mask=0,
                    end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0,
                    kernel_name="strided_slice_grad"):
    """
    verify the types of cast supported by tbe
    """
    strides_value = strides.get("const_value")
    if strides_value:
        for i in strides_value:
            if i != 1:
                return False, "strides has not 1 value."
    check_result = True, ""

    if (new_axis_mask != 0) or (shrink_axis_mask != 0 and
                                shrink_axis_mask != 2):
        reason = "the axis is not supported, new_axis_mask:%s, shrink_axis_mask:%s, shrink_axis_mask:%s"\
                  % (new_axis_mask, shrink_axis_mask, shrink_axis_mask)
        check_result = False, reason

    strides_size = strides.get("ori_shape")[0]
    shape_size = shape.get("ori_shape")[0]
    if shrink_axis_mask == 2 and (ellipsis_mask != 1 or
                                  strides_size != 2 or shape_size <= 2):
        reason = "the axis is not supported, new_axis_mask:%s, ellipsis_mask:%s, strides_size:%s, shape_size:%s"\
                  % (new_axis_mask, ellipsis_mask, strides_size, shape_size)
        check_result = False, reason

    return check_result


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
def strided_slice_grad(shape, begin, end, strides, dy, output, begin_mask=0,
                       end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0,
                       kernel_name="strided_slice_grad"):
    """ 
    Since `StridedSlice` cuts out pieces of its `input` which is size`shape_dy`, its gradient
    will have the same shape (which is passed here as `shape_x`). The gradient will be zero in any
    element that the slice does not select.

    Parameters
    ----------
    dy : dict
        shape and dtype of input
    output_x : dict
        shape and dtype of out
    shape : list or tuple.
        shape of input
    begin: list or tuple.
        represents the index of the first value to select.
    end: list or tuple.
        represents the index of the last value to select.
    strides: list or tuple.
        step length to select.
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
    pass
