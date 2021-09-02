#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR list_a PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

interp
"""
import te
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_build import build_config
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from .interp_common import GlobalParams
from .interp_in_hw_eq_out_hw import compute_with_in_hw_eq_out_hw
from .interp_in_hw_eq_one_fp16 import compute_with_in_hw_eq_one_fp16
from .interp_in_hw_eq_one_fp32 import compute_with_in_hw_eq_one_fp32
from .interp_out_hw_eq_one import compute_with_out_hw_eq_one
from .interp_normal import compute_with_in_normal_situation


def _interp_ir(inputs, outputs):
    """
        ir build part
        -------------
        Process:
            1. input H/W eq output H/W
            2. input H/W eq (1,1)
            3. output H/W eq (1,1)
            4. normal input/output H/W
    """
    para = GlobalParams(inputs, outputs)
    ib = tvm.ir_builder.create()

    # No.1 situation : input H/W == output H/W
    if para.h_in == para.h_out and para.w_in == para.w_out:
        ib = compute_with_in_hw_eq_out_hw(ib, para)
    # No.2 situation : input H/W == (1,1)
    elif para.h_in == 1 and para.w_in == 1:
        ib = compute_with_in_hw_eq_one_fp16(ib, para) if para.dtype == "float16" \
            else compute_with_in_hw_eq_one_fp32(ib, para)
    # No.3 situation : output H/W == (1,1)
    elif para.h_out == 1 and para.w_out == 1:
        ib = compute_with_out_hw_eq_one(ib, para)
    # No.4 normal input/output H/W
    else:
        ib = compute_with_in_normal_situation(ib, para)

    return ib.get()


@fusion_manager.register("interp")
def interp_compute(images, y, size, kernel_name):
    """interp schedule compute part

    Parameters
    ----------
    images: TVM tensor
        the placeholders of images value
    y: dict
        dict info of output value
    size: list
        the shape of output about 'new_height, new_width'
    kernel_name: str
        cce kernel name

    returns
    -------
    res
    """
    images_shape = shape_util.shape_to_list(images.shape)
    shape_out = list(images_shape)
    shape_out[-2] = size[-1]
    shape_out[-3] = size[-2]
    para_check.check_tensor_shape_size(shape_out)

    res = tvm.extern(tuple(shape_out), [images], lambda ins, outs: _interp_ir(ins[0], outs[0]), name="res",
                     dtype="float32")
    return res


@para_check.check_input_type(dict, dict, int, int, int, int, int, int, str)
def interp(images, y, height, width, zoom_factor, shrink_factor, pad_beg, pad_end, kernel_name="interp"):
    """interp schedule main part

    Parameters
    ----------
    images: dict
        dict info of images value, must include the keys(shape and dtype).
        and shape will be 5HD
    y: dict
        dict info of output value
    height: int
        the height of output
    width: int
        the width of output
    zoom_factor: int
        reserved
    shrink_factor: int
        reserved
    pad_beg: int
        reserved
    pad_end: int
        reserved
    kernel_name: str
        cce kernel name, default value is "interp"

    returns
    -------
    None
    """
    para_check.check_kernel_name(kernel_name)
    image_dtype = images.get("dtype")
    image_shape = images.get("shape")

    para_check.check_shape_rule(image_shape)
    check_list = ["float16", "float32"]
    if image_dtype not in check_list:
        raise RuntimeError("only support %s while dtype is %s" % (str(check_list), image_dtype))
    if len(image_shape) != 5:
        raise RuntimeError("The ndim of input must be 5," " while input ndim is %d" % (len(image_shape)))

    size = [height, width]
    para_check.check_shape_rule(size)
    if (image_shape[2] > 2048 or image_shape[3] > 2048) or (size[0] > 2048 or size[1] > 2048):
        raise RuntimeError("in or out h/w size should not larger than 2048")

    para_check.check_tensor_shape_size(image_shape)

    image_data = tvm.placeholder(image_shape, dtype=image_dtype, name="image_data")
    res = interp_compute(image_data, y, size, kernel_name)

    # build & output
    sch = tvm.create_schedule(res.op)
    with build_config:
        tvm.build(sch, [image_data, res], "cce", name=kernel_name)
