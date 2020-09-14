"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

segment_mean
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

SHAPE_SIZE_LIMIT = 2147483648


@fusion_manager.register("segment_mean")
def segment_mean_compute(input_x, input_y, output_y, kernel_name="segment_mean"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    input_y : list int
        the ids of input_tensor
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "segment_mean"

    Returns
    -------
    res : output of the data`s segment_mean, shape is (input_y[-1]+1, XXX),
          XXX is the same as input_x shape from the second dimension to the end dimension
    """

    res = te.lang.cce.unsorted_segment_mean(input_x, input_y, input_y[-1] + 1)
    return res


def ids_is_1d_and_sorted(ids):
    n = len(ids)
    if ids != sorted(ids):
        raise RuntimeError("ids must be sorted!")
    for i in ids:
        if not isinstance(i, int):
            raise RuntimeError("ids must be 1D and all int!")
        if i < 0:
            raise RuntimeError("ids should greater than zero!")


@util.check_input_type(dict, list, dict, str)
def segment_mean(input_tensor, segment_ids, output_y, kernel_name="segment_mean"):

    """
    calculating data

    Parameters
    ----------
    input_tensor : dict
        shape and dtype of first input, only support float16, float32
    segment_ids : list int
        shape and dtype of second input, only support int32
        size is equal to the size of input_data first dimension
        values should be sorted and can be repeated
    output_y : dict
        shape and dtype of output,
    kernel_name : str
        kernel name, default value is "segment_mean"

    Returns
    -------
    None
    """

    shape_tensor = input_tensor.get("shape")
    shape_segment = (len(segment_ids),)
    dtype_tensor = input_tensor.get("dtype")
    input_tensor_dtype = dtype_tensor.lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_tensor)
    util.check_shape_rule(shape_segment)
    util.check_tensor_shape_size(shape_tensor)
    util.check_tensor_shape_size(shape_segment)
    check_tuple_tensor = ("float16", "float32", "int32", "int8", "uint8")
    util.check_dtype_rule(dtype_tensor, check_tuple_tensor)


    # judgement of ids
    length_ids = len(segment_ids)
    if length_ids != shape_tensor[0]:
        raise RuntimeError("length of ids must equal to shape[0] of input_tensor!")
    ids_is_1d_and_sorted(segment_ids)

    if dtype_tensor == "int8":
        data_input = tvm.placeholder(shape_tensor, name="data_input", dtype=dtype_tensor)
        data_input1 = te.lang.cce.cast_to(data_input, "float16")
        res1 = segment_mean_compute(data_input1, segment_ids, output_y, kernel_name)
        res = te.lang.cce.cast_to(res1, "int8")
    else:
        data_input = tvm.placeholder(shape_tensor, name="data_input", dtype=dtype_tensor)
        res = segment_mean_compute(data_input, segment_ids, output_y, kernel_name)


    data_input = tvm.placeholder(shape_tensor, name="data_input", dtype=input_tensor_dtype)
    res = segment_mean_compute(data_input, segment_ids, output_y, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}

    te.lang.cce.cce_build_code(schedule, config)