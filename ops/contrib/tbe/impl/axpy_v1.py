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

axpy_v1
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


@fusion_manager.register("axpy_v1")
def axpy_v1_compute(input_a, input_x, input_y, output, kernel_name="axpy_v1"):
    """
    calculating data

    Parameters
    ----------
    input_a : TVM tensor
        the placeholder of input_a
    input_x : TVM tensor
        the placeholder of input_x
    input_y : TVM tensor
        the placeholder of input_y
    output : dict
        dict of output, include keys
    kernel_name : str
        kernel name, default value is "axpy_v1"

    Returns
    -------
    output tensor
    """
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)

    input_a = te.lang.cce.broadcast(input_a, shape_x)
    res = te.lang.cce.vmla(input_a, input_x, input_y)

    return res


@util.check_input_type(dict, dict, dict, dict, str)
def axpy_v1(input_a, input_x, input_y, output, kernel_name="axpy_v1"):
    """
    calculating data

    Parameters
    ----------
    input_a : dict
        shape and dtype of input
    input_x : dict
        shape and dtype of input
    input_y : dict
        shape and dtype of input

    output : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "axpy_v1"

    Returns
    -------
    None
    """

    shape_a = input_a.get("shape")  # get input shape and dtype of Axpy
    dtype_a = input_a.get("dtype")
    ori_shape_a = input_a.get("ori_shape")

    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    ori_shape_x = input_x.get("ori_shape")

    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype")
    ori_shape_y = input_y.get("ori_shape")

    input_dtype_a = dtype_a.lower()
    input_dtype_x = dtype_x.lower()
    input_dtype_y = dtype_y.lower()

    util.check_shape_rule(shape_a)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_a)  # check the size of tensor shape
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_y)
    util.check_kernel_name(kernel_name)  # check kernel_name

    if len(ori_shape_a) != 4 and len(ori_shape_a) != 2:
        raise RuntimeError("input_a should be 2D or 4D")
    if len(ori_shape_a) == 4 and ((shape_a[2] != 1) | (shape_a[3] != 1)):
        raise RuntimeError("dim H and W of input_a should be 1")
    if (dtype_a != dtype_x) | (dtype_x != dtype_y):
        raise RuntimeError("dtype of inputs should be consistent")
    if (ori_shape_a[0] != ori_shape_x[0]) | (ori_shape_a[1] != ori_shape_x[1]):
        raise RuntimeError("dim N and C of input_a and input_x should be consistent")
    if ori_shape_x != ori_shape_y:
        raise RuntimeError("shape of input_x and input_y should be consistent")

    dtype = dtype_a
    check_tuple = ("float16",)

    util.check_dtype_rule(dtype, check_tuple)

    data_input_a = tvm.placeholder(shape_a, name="data_input_a", dtype=input_dtype_a)
    data_input_x = tvm.placeholder(shape_x, name="data_input_x", dtype=input_dtype_x)
    data_input_y = tvm.placeholder(shape_y, name="data_input_y", dtype=input_dtype_y)

    res = axpy_v1_compute(data_input_a, data_input_x, data_input_y, output, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    # operator build
    config = {"print_ir": False,
              "need_build": True,
              "name": kernel_name,
              "tensor_list": [data_input_a, data_input_x, data_input_y, res]}

    te.lang.cce.cce_build_code(schedule, config)
