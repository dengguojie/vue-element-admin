#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this
file except in compliance with the License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0
ones_like
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as functools_reduce
from te.utils.op_utils import *

# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
@fusion_manager.register("ones_like")
def ones_like_compute(input_x, output_y, kernel_name="ones_like"):
    """
    Given a tensor, this operation returns a tensor of the same
    type and shape as `tensor` with all elements set to 1.

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: TVM tensor
        the placeholder of output data
    kernel_name : str
        cce kernel name, default value is "ones_like"

    Returns
    -------
    res: TVM tensor
        the result of ones_like_compute
    """
    src_dtype = input_x.dtype.lower()
    dst_type = src_dtype
    src_type_list = ("int8", "uint8")
    dst_type_list = ("int8", "uint8")
    if src_dtype in src_type_list:
        src_dtype = "float16"
    one = tvm.const(1, dtype=src_dtype)
    one_src = te.lang.cce.broadcast(one, input_x.shape)
    if src_dtype in dst_type_list:
        one_src = te.lang.cce.cast_to(one_src, dst_type,
                                       f1628IntegerFlag=True)
    else:
        one_src = te.lang.cce.cast_to(one_src, dst_type)
    with tvm.tag_scope("elewise_binary_phony"):
        res = te.tvm.compute(input_x.shape,
                             lambda *indices: one_src[indices] + input_x[indices],
                             name="elewise_binary_phony_output")

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def ones_like(x, y, kernel_name="ones_like"):
    """
    output a tensor of all one, shape and dtype is same of input

    Parameters
    ----------
    x: dict
        shape and dtype of input, only support float16, float32,
        int32,int8,uint8
    y: dict
        shape and dtype of output data
    kernel_name: str
        cce kernel name, default value is "ones_like"

    Returns
    ------
    None
    """
    shape_input = x.get("shape")
    dtype_input = x.get("dtype")
    check_shape(shape_input, param_name="x")
    check_list_src = ("float16", "float32", "int32", "int8", "uint8")
    src_dtype = dtype_input.lower()
    check_dtype(src_dtype, check_list_src, param_name="x")
    shape_input = (functools_reduce(lambda x, y: x * y, shape_input[:]),)
    input_x = tvm.placeholder(shape_input, name="input_x", dtype=src_dtype)
    res = ones_like_compute(input_x, y,
                            kernel_name=kernel_name)
    with tvm.target.cce():
        auto_sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_x, res]}
    te.lang.cce.cce_build_code(auto_sch, config)
