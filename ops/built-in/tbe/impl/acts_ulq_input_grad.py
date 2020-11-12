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

acts_ulq_input_grad
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648

# pylint: disable=locally-disabled,too-many-arguments,unused-argument

def acts_ulq_input_grad_compute(data_y_grad, data_clamp_min_mask, data_clamp_max_mask, kernel_name):
    """
    calculating grad of acts_ulq

    Parameters
    ----------
    data_y_grad: TVM tensor
        input grad
    data_clamp_min_mask: TVM tensor
        indicator where x > clamp_min
    data_clamp_max_mask: TVM tensor
        indicator where x < clamp_max
    kernel_name: str
        cce kernel name, default value is add

    Returns
    -------
    res : grade of acts_ulq
    """
    dtype = data_y_grad.dtype
    zero = te.lang.cce.broadcast(tvm.const(0, dtype), data_y_grad.shape)
    one = te.lang.cce.broadcast(tvm.const(1, dtype), data_y_grad.shape)
    signal = te.lang.cce.vsel(data_clamp_min_mask, one, zero)
    signal = te.lang.cce.vsel(data_clamp_max_mask, signal, zero)
    x_grad = te.lang.cce.vmul(data_y_grad, signal)

    return [x_grad]


@util.check_input_type(dict, dict, dict, dict, str)
def acts_ulq_input_grad(y_grad, clamp_min_mask, clamp_max_mask, x_grad, kernel_name="acts_ulq_input_grad"):
    """
    calculating grad of acts_ulq

    Parameters
    ----------
    data_y_grad: TVM tensor
        input grad
    data_clamp_min_mask: TVM tensor
        indicator where x > clamp_min
    data_clamp_max_mask: TVM tensor
        indicator where x < clamp_max
    kernel_name: str
        cce kernel name, default value is acts_ulq_input_grad

    Returns
    -------
    None
    """
    y_grad_shape = util.scalar2tensor_one(y_grad.get("shape"))
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(y_grad_shape)
    util.check_shape_size(y_grad_shape, SHAPE_SIZE_LIMIT)

    check_tuple = ("float16", "float32")
    y_grad_type = y_grad.get("dtype").lower()
    util.check_dtype_rule(y_grad_type, check_tuple)

    shape_clamp_min_mask = util.scalar2tensor_one(clamp_min_mask.get("shape"))
    shape_clamp_max_mask = util.scalar2tensor_one(clamp_max_mask.get("shape"))

    if y_grad_shape != shape_clamp_min_mask or y_grad_shape != shape_clamp_max_mask:
        raise ValueError("clamp max/min mask shape should be same as y_grad")

    clamp_min_mask_type = clamp_min_mask.get("dtype").lower()
    clamp_max_mask_type = clamp_max_mask.get("dtype").lower()

    if clamp_min_mask_type == "int8":
        clamp_min_mask_type = "bool"

    if clamp_max_mask_type == "int8":
        clamp_max_mask_type = "bool"

    if clamp_min_mask_type != "bool" or clamp_max_mask_type != "bool":
        raise ValueError("clamp min/max should be type bool")

    util.check_shape_size(y_grad_shape, SHAPE_SIZE_LIMIT)
    data_y_grad = tvm.placeholder(y_grad_shape, y_grad_type, 'data_y_grad')
    data_clamp_min_mask = tvm.placeholder(shape_clamp_max_mask, clamp_min_mask_type, 'data_clamp_min_mask')
    data_clamp_max_mask = tvm.placeholder(shape_clamp_max_mask, clamp_min_mask_type, 'data_clamp_max_mask')

    res = acts_ulq_input_grad_compute(data_y_grad, data_clamp_min_mask, data_clamp_max_mask, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)
    tensor_list = [data_y_grad, data_clamp_min_mask, data_clamp_max_mask] + list(res)
    config = {
              "bool_storage_as_1bit": False,
              "print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(schedule, config)
