#!/usr/bin/python3
# -*- coding: UTF-8 -*-
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

Reduction
"""
from __future__ import absolute_import
from functools import reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util
from te.utils.op_utils import *
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# pylint: disable=redefined-outer-name, too-many-arguments, E1101
def op_select_format(input_x, output_y, operation=1, axis=0, coeff=1.0, kernel_name="reduction"):
    """
    support to 5HD format
    Parameters
    ----------
    input_x : input tensor
    output_y: output tensor
    operation : can only be one of "1:SUM, 2:ASUM (sum of abs), 3:SUMSQ (sum of sqr), 4:MEAN"
    axis : the first axis to reduce, may be negative to index from the end
            (e.g., -1 for the last axis).If axis == 0, the output Blob always has
            the empty shape (count 1), performing reduction across the entire input.
    coeff : scale for output
    kernel_name : cce kernel name, default value is "cce_reductionLayer"
    Returns
    -------
    param_dynamic_in_json
    """
    input_ori_shape = input_x.get("ori_shape")
    input_ori_format = input_x.get("ori_format")

    if axis < 0:
        axis = len(input_ori_shape) + axis

    is_support_5hd = True

    if input_ori_format not in ("NCHW", "NHWC"):
        is_support_5hd = False

    if (input_ori_format == "NCHW" and axis == 1) \
            or (input_ori_format == "NHWC" and axis == 3):
        is_support_5hd = False

    if len(input_ori_shape) < 4:
        is_support_5hd = False

    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES", "Hi3796CV300CS"):
        dtype_base = ["float16"]
    else:
        dtype_base = ["float16", "float32"]

    format_base = ["ND"] * len(dtype_base)
    if is_support_5hd:
        dtype_base = dtype_base + ["float16"]
        format_base = format_base + ["NC1HWC0"]

    dtype_base = ','.join(dtype_base)
    format_base = ','.join(format_base)

    input0 = gen_param(
        classify="input0", name="x", datatype=dtype_base, format=format_base)
    output0 = gen_param(
        classify="output0", name="y", datatype=dtype_base, format=format_base)
    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@fusion_manager.register("reduction")
def reduction_compute(data_info, product_verion, operation, axis, coeff):
    """
    Reduce a tensor on a certain axis, and scale output with coeff
    Parameters
    ----------
    data_info: include TVM tensor,shape and dtype
    product_verion: include mini("1.1"、"1.3"),cloud("1.6"),es("5.10"),1951("2.3")
    operation : can only be one of "1:SUM, 2:ASUM (sum of abs), 3:SUMSQ (sum of sqr), 4:MEAN"
    axis : the axis to reduce
    coeff : scale for output
    Returns
    -------
    output of the data's reduction
    """

    input_data = data_info.get("tensor")
    input_data_shape = data_info.get("shape")
    input_data_dtype = data_info.get("dtype")

    if product_verion not in ("Hi3796CV300ES", "Hi3796CV300CS"):
        if input_data_dtype == "float16":
            input_data = te.lang.cce.cast_to(input_data, "float32")

    # computational process
    if operation == 2:
        data_tmp_input = te.lang.cce.vabs(input_data)
        tmp = te.lang.cce.vmuls(data_tmp_input, coeff)

    elif operation == 3:
        data_tmp_input = te.lang.cce.vmul(input_data, input_data)
        tmp = te.lang.cce.vmuls(data_tmp_input, coeff)

    elif operation == 4:
        size = input_data_shape[-1]
        cof = float(coeff * (size ** (-0.5)))
        tmp = te.lang.cce.vmuls(input_data, cof)

    elif operation == 1:
        tmp = te.lang.cce.vmuls(input_data, coeff)

    res = te.lang.cce.sum(tmp, axis=axis)

    if operation == 4:
        size = input_data_shape[-1]
        size_reci = float(size ** (-0.5))
        res = te.lang.cce.vmuls(res, size_reci)

    if product_verion not in ("Hi3796CV300ES", "Hi3796CV300CS"):
        if input_data_dtype == "float16":
            res = te.lang.cce.cast_to(res, "float16")

    return res


# pylint: disable=redefined-outer-name, too-many-arguments, E1101
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_INT,
                 OPTION_ATTR_INT,OPTION_ATTR_FLOAT, KERNEL_NAME)
def reduction(input_x, output_y, operation=1, axis=0, coeff=1.0, kernel_name="reduction"):
    """
    Reduce a tensor on a certain axis, and scale output with coeff
    Parameters
    ----------
    input_x : input tensor
    output_y: output tensor
    operation : can only be one of "1:SUM, 2:ASUM (sum of abs), 3:SUMSQ (sum of sqr), 4:MEAN"
    axis : the first axis to reduce, may be negative to index from the end
            (e.g., -1 for the last axis).If axis == 0, the output Blob always has
            the empty shape (count 1), performing reduction across the entire input.
    coeff : scale for output
    kernel_name : cce kernel name, default value is "cce_reductionLayer"
    Returns
    -------
    None
    """
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    # input_x's shape check
    ori_shape = list(input_x.get("ori_shape"))
    check_shape(ori_shape, param_name="input_x")

    # input_x' dtype check
    inp_dtype = input_x.get("dtype").lower()
    check_dtype(inp_dtype, ("float16", "float32"), param_name="input_x")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS") and inp_dtype == "float32":
        error_Info = {}
        error_Info['errCode'] = 'E81006'
        error_Info['param_name'] = 'dtype'
        error_Info['op_name'] = 'reduction'
        error_Info['real_value'] = inp_dtype
        raise RuntimeError("In op[%s], ES is not supported while the [%s] of input is [%s]."
                           %(error_Info['op_name'],error_Info['param_name'],error_Info['real_value']))

    # axis parameter check
    if axis >= len(ori_shape) or axis < -len(ori_shape):
        error_Info = {}
        error_Info['errCode'] = OP_ERROR_CODE_002
        error_Info['param_name'] = 'axis'
        error_Info['op_name'] = 'reduction'
        error_Info['min_value'] = -len(ori_shape)
        error_Info['max_value'] = len(ori_shape)-1
        error_Info['real_value'] = axis
        raise RuntimeError(error_Info,"In op[%s], the parameter[%s] should be "
                                      "in the range of (%s,%s), but actually is [%s]."
                           %(error_Info['op_name'],error_Info['param_name'], \
                             error_Info['min_value'],error_Info['max_value'],error_Info['real_value']))

    # operation parameter check
    if operation not in (1, 2, 3, 4):
        error_Info = {}
        error_Info['errCode'] = 'E80002'
        error_Info['param_name'] = 'operation'
        error_Info['op_name'] = 'reduction'
        error_Info['expect_value_list'] = (1, 2, 3, 4)
        error_Info['real_value'] = operation
        raise RuntimeError(error_Info,"In op[%s], the parameter[%s] should "
                                      "only be one of [%s], but actually is [%s]."
                           %(error_Info['op_name'],error_Info['param_name'], \
                             error_Info['expect_value_list'],error_Info['real_value']))

    # Preprocess
    if axis < 0:
        axis = len(ori_shape) + axis

    shape = list(input_x.get("shape"))

    if input_x.get("format") == "NC1HWC0":
        axis = util.axis_transfrom_5d(axis, input_x.get("ori_format"))
        if axis > 1:
            shape = shape[:axis] + [reduce(lambda x, y: x * y, shape[axis:-1])] + [shape[-1]]
        if axis == 1:
            raise RuntimeError("The C axis does not support reduction when the data format is NC1HWC0.")
        if axis == 0:
            shape = [reduce(lambda x, y: x * y, shape)]
    else:
        shape = shape[:axis] + [reduce(lambda x, y: x * y, shape[axis:])]

    # define input
    data = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)
    data_info = {"tensor": data, "shape": shape, "dtype": inp_dtype}

    res = reduction_compute(data_info, cce_product, operation, axis, coeff)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}
    te.lang.cce.cce_build_code(sch, config)
