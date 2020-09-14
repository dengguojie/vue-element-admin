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

sigmoid_grad
"""
import operator
from functools import reduce as reduce_ins
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
import te.lang.dynamic
import topi
from topi import generic
from functools import reduce as reduceIns
from te.platform.shape_classifier import classify, Mode
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import REQUIRED_ATTR_LIST_INT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import variable_shape
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import check_elewise_shape_range

# General limitation of the reduce size for input shape: 2**30
SHAPE_SIZE_LIMIT = 2 ** 30


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def sigmoid_grad_compute(x, y, z, kernel_name="sigmoid_grad"):
    """
    algorithm : sigmoid grad compute

    sigmoid_grad = (sigmoid - sigmoid*sigmoid)*grad

    Parameters:
    ----------
    x : a tensor of input data

    y : a tensor of grad

    z : output dict

    kernel_name : cce kernel name, default value is "sigmoid_grad"

    Returns
    -------
    a tenosr
    """
    dtype = x.dtype.lower()
    cast_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.cast_to", "f322f16")
    if dtype == "float32" and not cast_support:
        raise RuntimeError(
            "float32 transfer to float16 is only supported on mini and cloud platform")
    vmul_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmul", "float32")
    vsub_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vsub", "float32")
    if dtype == "float16":
        x = te.lang.dynamic.cast_to(x, "float32")
        y = te.lang.dynamic.cast_to(y, "float32")
    sigmoid_square = te.lang.dynamic.vmul(x, x)
    if dtype == "float32" and not vmul_support:
        sigmoid_square = te.lang.dynamic.cast_to(sigmoid_square, "float16")
    tensor_sub = te.lang.dynamic.vsub(x, sigmoid_square)
    if dtype == "float32" and not vsub_support:
        tensor_sub = te.lang.dynamic.cast_to(tensor_sub, "float16")
    res = te.lang.dynamic.vmul(tensor_sub, y)
    if dtype == "float16":
        res = te.lang.dynamic.cast_to(res, "float16")
    return res


@te.op.register_operator("SigmoidGrad")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def sigmoid_grad(x,
                 dx,
                 out,
                 kernel_name="sigmoid_grad"):
    """
    do sigmoid grad

    sigmoid_grad = (sigmoid - sigmoid*sigmoid)*grad

    Parameters:
    ----------
    x : dictionary shape of sigmoid input

    dx : dictionary shape of grad

    out: dictionary output

    kernel_name : cce kernel name, default value is "sigmoid_grad_cce"

    Returns
    -------
    None
    """
    x_dtype = x.get("dtype").lower()
    dx_dtype = dx.get("dtype").lower()
    check_list = ("float16", "float32")
    check_dtype(x_dtype, check_list, param_name="input_x")
    check_dtype(dx_dtype, check_list, param_name="input_dx")
    check_elewise_shape_range([x, dx])
    if x_dtype != dx_dtype:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_018
        error_info['op_name'] = 'sigmoid_grad'
        error_info['param_name1'] = 'x_dtype'
        error_info['param_name2'] = 'dx_dtype'
        error_info['param1_dtype'] = str(x_dtype)
        error_info['param2_dtype'] = str(dx_dtype)
        raise RuntimeError("In op[%s], the parameter[%s][%s] are not equal in "
                           "dtype with dtype[%s][%s]." % (
                               error_info['op_name'],
                               error_info['param_name1'],
                               error_info['param_name2'],
                               error_info['param1_dtype'],
                               error_info['param2_dtype']))

    ins = classify([x, dx], Mode.ELEWISE)
    schedules, tensors = [], []
    for (sig, d) in ins:
        with te.op.compute():
            sig_shape = variable_shape([sig])
            shape_sig = reduce_ins(lambda x, y: x * y, sig_shape[:])
            tensor_x = tvm.placeholder(shape_sig, x_dtype, "tensor_x")
            tensor_dx = tvm.placeholder(shape_sig, x_dtype, "tensor_dx")
            res = sigmoid_grad_compute(tensor_x, tensor_dx, out, kernel_name)
            tensors.append((tensor_x, tensor_dx, res))
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
