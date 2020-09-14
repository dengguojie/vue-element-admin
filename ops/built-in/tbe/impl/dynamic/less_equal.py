#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

less_equal
"""
import te.lang.dynamic
from te import platform as tbe_platform
from te import tvm
from te.platform.shape_classifier import classify, Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_elewise_shape_range
from te.utils.op_utils import variable_shape
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import OP_ERROR_CODE_018
from topi import generic

# define a scalar, value = 2**(-126), minimun num of float32 2**(-126)
SCALAR_MIN_FP32 = 2 ** (-126)
# define a scalar, value = 2**(50)
SCALAR_MUL_FP32 = 2 ** (50)
# define a scalar, value = 2**(26)
SCALAR_MUL2_FP32 = 2 ** (26)
# define a scalar, value = 2**(-24), minimun num of float16 2**(-24)
SCALAR_MIN_FP16 = 2 ** (-24)
# define a scalar, value = 2**(12)
SCALAR_MUL_FP16 = 2 ** (12)
# define a scalar, value = -1
SCALAR_NEG_ONE = -1


# pylint: disable=locally-disabled,unused-argument,too-many-locals
def less_equal_compute(input_x, input_y, output_z, kernel_name="less_equal"):
    """
    compute for less_equal

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict info of output_z
    kernel_name: str
        cce kernel name, default value is "less_equal"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype_x = input_x.dtype
    shape_x = te.lang.dynamic.shape_to_list(input_x.shape)
    shape_y = te.lang.dynamic.shape_to_list(input_y.shape)
    shape_x, shape_y, shape_broadcast = broadcast_shapes(shape_x, shape_y,
        param_name_input1="input_x", param_name_input2="input_y")

    if dtype_x == "float32":
        scalar_min = tvm.const(SCALAR_MIN_FP32, dtype="float32")
        scalar_mul = tvm.const(SCALAR_MUL_FP32, dtype="float32")
        scalar_mul1 = tvm.const(SCALAR_MUL2_FP32, dtype="float32")
        scalar_neg_one = tvm.const(SCALAR_NEG_ONE, dtype="float32")
    else:
        scalar_min = tvm.const(SCALAR_MIN_FP16, dtype="float16")
        scalar_mul = tvm.const(SCALAR_MUL_FP16, dtype="float16")
        scalar_neg_one = tvm.const(SCALAR_NEG_ONE, dtype="float16")

    if dtype_x in ("int8", "uint8"):
        input_x = te.lang.dynamic.cast_to(input_x, "float16")
        input_y = te.lang.dynamic.cast_to(input_y, "float16")

    input_x = te.lang.dynamic.broadcast(input_x, shape_broadcast)
    input_y = te.lang.dynamic.broadcast(input_y, shape_broadcast)

    res_max = te.lang.dynamic.vmax(input_x, input_y)
    res_vsub = te.lang.dynamic.vsub(input_y, res_max)
    if tbe_platform.cce_conf.api_check_support("te.lang.dynamic.vabs",
                                               res_vsub.dtype):
        res_vabs = te.lang.dynamic.vabs(res_vsub)
    else:
        res_vsub = te.lang.dynamic.cast_to(res_vsub, "float32")
        res_vabs = te.lang.dynamic.vabs(res_vsub)

    res_min = te.lang.dynamic.vmins(res_vabs, scalar_min)
    res_vmul = te.lang.dynamic.vmuls(res_min, scalar_mul)
    res_vmul1 = te.lang.dynamic.vmuls(res_vmul, scalar_mul)

    if dtype_x == "float32":
        res_vmul2 = te.lang.dynamic.vmuls(res_vmul1, scalar_mul1)
        res_vsub1 = te.lang.dynamic.vadds(res_vmul2, scalar_neg_one)
        res_vabs1 = te.lang.dynamic.vabs(res_vsub1)
    else:
        res_vsub1 = te.lang.dynamic.vadds(res_vmul1, scalar_neg_one)
        res_vabs1 = te.lang.dynamic.vabs(res_vsub1)

    res = te.lang.dynamic.cast_to(res_vabs1, "int8", True)

    return res


@te.op.register_operator("LessEqual")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def less_equal(input_x, input_y, output_z, kernel_name="less_equal"):
    """
    Returns the truth value of (x <= y) element-wise

    Parameters
    ----------
    input_x: dict
        dict{"shape":tuple or list, "dtype":str, range: tuple or list},
        shape, range, and dtype of first input,
        support float16,float32,int32,int8,uint8
    input_y: dict
        dict{"shape":tuple or list, "dtype":str, range: tuple or list},
        shape, range, and dtype of first input,
        support float16,float32,int32,int8,uint8
    output_z: dict
        dict of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is "less_equal"

    Returns
    -------
    None
    """
    # check input tensor data_type
    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_y.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "uint8", "int8")
    check_dtype(x_dtype, check_list, param_name="input_x")
    check_dtype(y_dtype, check_list, param_name="input_y")
    check_elewise_shape_range([input_x, input_y], support_broadcast=True)
    if x_dtype != y_dtype:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_018
        error_info['op_name'] = 'less_equal'
        error_info['param_name1'] = 'x_dtype'
        error_info['param_name2'] = 'y_dtype'
        error_info['param1_dtype'] = str(x_dtype)
        error_info['param2_dtype'] = str(y_dtype)
        raise RuntimeError(error_info,
                           "In op[%s], the parameter[%s][%s] are not equal in "
                           "dtype with dtype[%s][%s]." % (
                               error_info['op_name'],
                               error_info['param_name1'],
                               error_info['param_name2'],
                               error_info['param1_dtype'],
                               error_info['param2_dtype']))

    ins = classify([input_x, input_y], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input_x, input_y) in ins:
        with te.op.compute():
            # shape
            x_shape, y_shape = variable_shape([input_x, input_y],
                                              support_broadcast=True)
            x_shape, y_shape, shape_max = broadcast_shapes(x_shape, y_shape,
                param_name_input1="input_x", param_name_input2="input_y")
            x_shape, y_shape = refine_shapes_for_broadcast(x_shape, y_shape)

            # less_equal compute
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            tensor_y = tvm.placeholder(y_shape, y_dtype, "tensor_y")
            res = less_equal_compute(tensor_x, tensor_y, output_z, kernel_name)

            tensors.append([tensor_x, tensor_y, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
