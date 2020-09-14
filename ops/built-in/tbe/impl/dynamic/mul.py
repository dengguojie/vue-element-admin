#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

dynamic mul
"""
import te.lang.dynamic
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


def mul_compute(input1, input2, output, kernel_name="mul"):
    """
    calculating data's mul, c = a * b

    Parameters
    ----------
    input1: TVM tensor
        the placeholder of first input data
    input2: TVM tensor
        the placeholder of second input data
    output: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is mul

    Returns
    -------
    res : output of the data's mul
    """
    x0_shape = te.lang.dynamic.shape_to_list(input1.shape)
    x1_shape = te.lang.dynamic.shape_to_list(input2.shape)
    x0_shape, x1_shape, y_shape = broadcast_shapes(x0_shape, x1_shape,
                                                   param_name_input1="input1",
                                                   param_name_input2="input2")
    input1 = te.lang.dynamic.broadcast(input1, y_shape)
    input2 = te.lang.dynamic.broadcast(input2, y_shape)
    res = te.lang.dynamic.vmul(input1, input2)

    return res


@te.op.register_operator("Mul")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def mul(input1, input2, output, kernel_name="mul"):
    """
    algorithm: mul
    calculating data's mul, c = a * b

    Parameters
    ----------
    input1 : dict
        include ori_shape, shape, ori_format, format, dtype and range
        dtype only support float16, float32, int32
    input2 : dict
        include ori_shape, shape, ori_format, format, dtype and range
        dtype only support float16, float32, int32
    output: dict
        include ori_shape, shape, ori_format, format, dtype and range
        shape must be broadcast shape of input
    kernel_name : str
        cce kernel name, default value is mul

    Returns
    -------
    None
    """

    # check dtype
    dtype_x1 = input1.get("dtype").lower()
    dtype_x2 = input2.get("dtype").lower()
    check_list = ("float16", "float32", "int32")
    check_dtype(dtype_x1, check_list, param_name="input1")
    check_dtype(dtype_x2, check_list, param_name="input2")
    check_elewise_shape_range([input1, input1], support_broadcast=True)
    if dtype_x1 != dtype_x2:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_018
        error_info['op_name'] = 'mul'
        error_info['param_name1'] = 'dtype_x1'
        error_info['param_name2'] = 'dtype_x2'
        error_info['param1_dtype'] = str(dtype_x1)
        error_info['param2_dtype'] = str(dtype_x2)
        raise RuntimeError(error_info,
                           "In op[%s], the parameter[%s][%s] are not equal in "
                           "dtype with dtype[%s][%s]." % (
                               error_info['op_name'],
                               error_info['param_name1'],
                               error_info['param_name2'],
                               error_info['param1_dtype'],
                               error_info['param2_dtype']))

    ins = classify([input1, input2], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input1, input2) in ins:
        with te.op.compute():
            # shape
            shape_x1, shape_x2 = variable_shape([input1, input2],
                                                support_broadcast=True)
            shape_x1, shape_x2, shape_max = \
                broadcast_shapes(shape_x1, shape_x2,
                                 param_name_input1="input1",
                                 param_name_input2="input2")
            shape_x1, shape_x2 = refine_shapes_for_broadcast(shape_x1,
                                                             shape_x2)
            # mul_compute
            data_x1 = tvm.placeholder(shape_x1, dtype=dtype_x1, name="data_x1")
            data_x2 = tvm.placeholder(shape_x2, dtype=dtype_x2, name="data_x2")
            res = mul_compute(data_x1, data_x2, output, kernel_name)

            tensors.append((data_x1, data_x2, res))
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
