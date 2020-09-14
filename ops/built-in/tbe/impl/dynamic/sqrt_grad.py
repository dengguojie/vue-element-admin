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

sqrt_grad
"""

import operator
from functools import reduce as reduce_ins
import te.lang.dynamic
from te import platform as tbe_platform
from te import tvm
from impl.util import fusion_util
from topi.cce import util
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_elewise_shape_range
from te.platform.shape_classifier import classify, Mode
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.utils.op_utils import check_op_params
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import variable_shape
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import refine_shapes_for_broadcast

# General limitation of the reduce size for input shape: 2**30
SHAPE_LIMIT = 1 << 30


# pylint: disable=locally-disabled,too-many-arguments,unused-argument

def sqrt_grad_compute(x, dx, out, kernel_name="sqrt_grad"):
    """
    algorithm: sqrt_grad_compute
    output_grad = input_grad/(2*input)

    Parameters
    ----------
    x: a tensor of input data

    dx : a tensor of grad

    Returns
    -------
    output data

    """

    x_shape = te.lang.dynamic.shape_to_list(x.shape)
    dx_shape = te.lang.dynamic.shape_to_list(dx.shape)
    x_shape, y_shape, _shape = broadcast_shapes(x_shape, dx_shape,
                                                param_name_input1="input_x",
                                                param_name_input2="input_dx")
    input_x = te.lang.dynamic.broadcast(x, _shape)
    input_dx = te.lang.dynamic.broadcast(dx, _shape)

    dtype = x.dtype.lower()
    mul_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        raise RuntimeError(
            "Input dtype only support float16 while input dtype is float32")
    const_val_2 = tvm.const(2.0, dtype)
    mul_val = te.lang.dynamic.vmuls(input_x, const_val_2)
    res = te.lang.dynamic.vdiv(input_dx, mul_val)
    return res


@te.op.register_operator("SqrtGrad")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def sqrt_grad(x, dx, out, kernel_name="sqrt_grad"):
    """
    algorithm: sqrt_grad_cce

    Parameters
    ----------
    x : dict of data: dict

    dx : dict of data_grad: dict

    out : dict of output: dict

    kernel_name : cce kernel name, default value is "sqrt_grad": str

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
        error_info['op_name'] = 'sqrt_grad'
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

    ins = classify([x, dx], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (x, dx) in ins:
        with te.op.compute():
            x_shape, dx_shape = variable_shape([x, dx], support_broadcast=True)
            x_shape, dx_shape, _ = broadcast_shapes(x_shape, dx_shape,
                                                    param_name_input1="input_x",
                                                    param_name_input2="input_dx")
            x_shape, dx_shape = refine_shapes_for_broadcast(x_shape, dx_shape)
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            tensor_dx = tvm.placeholder(dx_shape, dx_dtype, "tensor_dx")
            res = sqrt_grad_compute(tensor_x, tensor_dx, out, kernel_name)
            tensors.append((tensor_x, tensor_dx, res))
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
