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


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
def check_supported(shape, begin, end, strides, dy, output, begin_mask=0,
                    end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0,
                    kernel_name="strided_slice_grad_d"):
    """
    the value of new_axis_mask should in {0, 2, 4, 6}
    the value of shrink_axis_mas should in {0, 1, 2, 4}
    the length of shape should not equal to 0.
    """
    begin_value, end_value, strides_value = begin.get("const_value"), end.get("const_value"), \
                                            strides.get("const_value")
    if not begin_value or not end_value or not strides_value:
        return False, "begin and end and strides can not be const."

    if strides_value:
        for i in strides_value:
            if i != 1:
                return False, "strides has not 1 value."
    check_result = True, ""

    supported_new_axis_mask = {0, 2, 4, 6}
    supported_shrink_axis_mask = {0, 1, 2, 4}

    if new_axis_mask not in supported_new_axis_mask:
        reason = "the new_axis_mask is not supported, new_axis_mask:%s, supported_new_axis_mask:%s" \
                 % (new_axis_mask, supported_new_axis_mask)
        check_result = False, reason

    if shrink_axis_mask not in supported_shrink_axis_mask:
        reason = "the shrink_axis_mask is not supported, shrink_axis_mask:%s, supported_shrink_axis_mask:%s" \
                 % (shrink_axis_mask, supported_shrink_axis_mask)
        check_result = False, reason

    if "const_value" in shape.keys() and not shape.get("const_value"):
        reason = "should not be empty shape, shape:%s" % str(shape)
        check_result = False, reason

    return check_result


# 'pylint: disable=locally-disabled,too-many-arguments,invalid-name
@register_operator("StridedSliceGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
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

    obj = PadInit(kernel_name)
    obj.init_src_dst_gm((shape, begin, end, strides, dy), (output,), pad_input_idx=4, pad_outnput_idx=0)

    outer_compile = dict()
    outer_compile["begin_mask"] = begin_mask
    outer_compile["end_mask"] = end_mask
    outer_compile["ellipsis_mask"] = ellipsis_mask
    outer_compile["new_axis_mask"] = new_axis_mask
    outer_compile["shrink_axis_mask"] = shrink_axis_mask
    return obj.pad_compute(outer_compile)
