#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

acts_ulq
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util

# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
EPS = 1.192092896e-07


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-locals,invalid-name
def acts_ulq_compute(data, clamp_min, clamp_max, fixed_min, step, kernel_name):
    """
    calculating data's add, c = a + b

    Parameters
    ----------
    data. TVM tensor
        the placeholder of input data
    clamp_min: TVM tensor
        the placeholder of clamp min
    clamp_max: TVM tensor
        the placeholder of clamp max
    fixed_min: bool
        attr, indicate whether fix clamp min to zero
    kernel_name: str
        cce kernel name, default value is add

    Returns
    -------
    res : output tensors
    """
    if fixed_min:
        ori_clip_min = te.lang.cce.vmuls(clamp_min, tvm.const(0, clamp_min.dtype))
    else:
        # forcing pass zero
        ori_clip_min = te.lang.cce.vmins(clamp_min, tvm.const(0, clamp_min.dtype))

    ori_clip_max = te.lang.cce.vmaxs(clamp_max, tvm.const(step*EPS, clamp_max.dtype))

    scale = te.lang.cce.vsub(ori_clip_max, ori_clip_min)

    scale = te.lang.cce.vdiv(scale, te.lang.cce.broadcast(tvm.const(step, scale.dtype), scale.shape))

    offset = te.lang.cce.vdiv(ori_clip_min, scale)
    offset = te.lang.cce.round(offset)
    offset = te.lang.cce.cast_to(offset, data.dtype)

    #fake quant clip min/max
    clip_min = te.lang.cce.vmul(scale, offset)
    clip_max = te.lang.cce.vadds(offset, tvm.const(step, offset.dtype))
    clip_max = te.lang.cce.vmul(clip_max, scale)

    #clip data = data
    clamped_x = te.lang.cce.vmax(data, te.lang.cce.broadcast(clip_min, data.shape))
    clamped_x = te.lang.cce.vmin(clamped_x, te.lang.cce.broadcast(clip_max, data.shape))

    #adjust shape first
    clamp_min_mask = te.lang.cce.vcmp(data, te.lang.cce.broadcast(clip_min, data.shape), 'ge')
    clamp_max_mask = te.lang.cce.vcmp(data, te.lang.cce.broadcast(clip_max, data.shape), 'le')

    #fake quant x
    raw_x = te.lang.cce.vdiv(clamped_x, te.lang.cce.broadcast(scale, clamped_x.shape))
    round_x = te.lang.cce.round(raw_x)
    round_x = te.lang.cce.cast_to(round_x, data.dtype)

    clamped_loss = te.lang.cce.vsub(round_x, raw_x)
    clamped_loss = te.lang.cce.vdiv(
        clamped_loss,
        te.lang.cce.broadcast(tvm.const(step, scale.dtype), clamped_loss.shape))

    raw_m = te.lang.cce.vdiv(ori_clip_min, scale)
    round_m = te.lang.cce.round(raw_m)
    round_m = te.lang.cce.cast_to(round_m, data.dtype)
    loss_m = te.lang.cce.vsub(round_m, raw_m)
    loss_m = te.lang.cce.vdiv(loss_m, te.lang.cce.broadcast(tvm.const(step, loss_m.dtype), loss_m.shape))
    clamped_loss = te.lang.cce.vsel(clamp_min_mask, clamped_loss, te.lang.cce.broadcast(loss_m, clamped_x.shape))
    clamped_loss = te.lang.cce.vsel(clamp_max_mask, clamped_loss, te.lang.cce.broadcast(loss_m, clamped_x.shape))

    output = te.lang.cce.vmul(round_x, te.lang.cce.broadcast(scale, clamped_x.shape))

    return [output, clamp_min_mask, clamp_max_mask, clamped_loss]


@para_check.check_input_type(dict, dict, dict, dict, dict, dict, dict, bool, int, str)
def acts_ulq(
        data,
        clamp_min,
        clamp_max,
        output,
        clamp_min_mask,
        clamp_max_mask,
        x_clamped_loss,
        fixed_min,
        num_bits,
        kernel_name="acts_ulq"):
    """
    algorithm: ulq

    Parameters
    ----------
    data: dict
        shape and dtype of feature map, only support float16, float32
    clamp_min: dict
        shape and dtype of clamp min, only support float16, float32
    clamp_max: dict
        shape and dtype of clamp max, only support float16, float32
    y: dict
        shape and dtype of output
    clamp_min_mask:
        mask if data > clamp_min (fake quant)
    clamp_max_mask:
        mask if data < clamp_max (fake quant)
    x_clamped_loss:
        loss
    kernel_name : str
        cce kernel name, default value is acts_ulq

    Returns
    -------
    None
    """
    if num_bits != 8:
        raise ValueError("num bits only supports 8")
    shape_x = shape_util.scalar2tensor_one(data.get("shape"))
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_x)
    para_check.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)

    check_tuple = ("float16", "float32")
    input_data_type = data.get("dtype").lower()
    para_check.check_dtype_rule(input_data_type, check_tuple)

    shape_clamp_min = shape_util.scalar2tensor_one(clamp_min.get("shape"))
    shape_clamp_max = shape_util.scalar2tensor_one(clamp_max.get("shape"))

    if len(shape_clamp_min) != len(shape_clamp_max):
        raise ValueError("clamp min shape must be the same as clamp max")

    clamp_len = len(shape_clamp_min)

    if clamp_len != len(shape_x):
        raise ValueError("clamp min/max should be same dims as data")
    for i in range(clamp_len):
        if shape_clamp_min[i] != 1 or shape_clamp_max[i] != 1:
            raise ValueError("clamp_min/max should be all one")

    clamp_min_type = clamp_min.get("dtype").lower()
    clamp_max_type = clamp_max.get("dtype").lower()

    if clamp_min_type != input_data_type or clamp_max_type != input_data_type:
        raise ValueError("clamp max/min data type should be same as data")

    data_x = tvm.placeholder(shape_x, name="data_x", dtype=input_data_type)
    data_clamp_min = tvm.placeholder(shape_clamp_min, name="data_clamp_min", dtype=input_data_type)
    data_clamp_max = tvm.placeholder(shape_clamp_max, name="data_clamp_max", dtype=input_data_type)


    n = 2**num_bits - 1
    res = acts_ulq_compute(data_x, data_clamp_min, data_clamp_max, fixed_min, n, kernel_name)

    with tvm.target.cce():
        schedule = te.lang.cce.auto_schedule(res)
    tensor_list = [data_x, data_clamp_min, data_clamp_max] + list(res)
    config = {"print_ir": False,
              "bool_storage_as_1bit": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(schedule, config)
