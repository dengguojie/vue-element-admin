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

dynamic bn_training_update_v3
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util import fusion_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from tbe.dsl.base.operation import add_compile_info


# 'pylint: disable=redefined-builtin
def _check_shape_5hd(shape_x, shape_sum, shape_square_sum,
                     shape_scale, shape_offset):
    if len(shape_x) != 5 or len(shape_sum) != 5 \
            or len(shape_square_sum) != 5 or len(shape_scale) != 5 \
            or len(shape_offset) != 5:
        param_name = ['len(shape_x), len(shape_sum), len(shape_square_sum), len(shape_scale), len(shape_offset)']
        real_value = [len(shape_x), len(shape_sum), len(shape_square_sum), len(shape_scale), len(shape_offset)]
        error_manager_vector.raise_err_input_value_invalid("bn_training_update_v3", param_name, 5, real_value)
    dim_c1 = shape_x[1]
    dim_c0 = shape_x[4]

    if shape_sum[1] != dim_c1 or shape_sum[4] != dim_c0:
        error_manager_vector.raise_err_input_value_invalid("bn_training_update_v3", "shape_sum[1], shape_sum[4]",
        str(dim_c1) + ", " + str(dim_c0), str(shape_sum[1]) + ", " + str(shape_sum[4]))
    if shape_square_sum[1] != dim_c1 or shape_square_sum[4] != dim_c0:
        error_manager_vector.raise_err_input_value_invalid("bn_training_update_v3", "shape_square_sum[1], \
        shape_square_sum[4]",
        str(dim_c1) + ", " + str(dim_c0), str(shape_square_sum[1]) + ", " + str(shape_square_sum[4]))
    if shape_scale[1] != dim_c1 or shape_scale[4] != dim_c0:
        error_manager_vector.raise_err_input_value_invalid("bn_training_update_v3", "shape_scale[1], shape_scale[4]",
        str(dim_c1) + ", " + str(dim_c0), str(shape_scale[1]) + ", " + str(shape_scale[4]))
    if shape_offset[1] != dim_c1 or shape_offset[4] != dim_c0:
        error_manager_vector.raise_err_input_value_invalid("bn_training_update_v3", "shape_offset[1], shape_offset[4]",
        str(dim_c1) + ", " + str(dim_c0), str(shape_offset[1]) + ", " + str(shape_offset[4]))


def _check_dtype(dtype_x, dtype_sum, dtype_square_sum,
                 dtype_scale, dtype_offset):
    para_check.check_dtype(dtype_x, ("float16", "float32"))
    para_check.check_dtype(dtype_sum, ("float32",))
    para_check.check_dtype(dtype_square_sum, ("float32",))
    para_check.check_dtype(dtype_scale, ("float32",))
    para_check.check_dtype(dtype_offset, ("float32",))


# 'pylint: disable=too-many-nested-blocks
def _refine_ins_list(ins_list):
    for index, ins_list_value in enumerate(ins_list):
        shape_range = []
        for dim, dim_val in enumerate(ins_list[index]["shape"]):
            if dim_val == -1:
                if "range" in ins_list_value:
                    range_bottom, range_top = ins_list[index]["range"][dim]
                    if range_bottom <= 1:
                        if range_top is not None and range_top <= 1:
                            range_top = 2
                        shape_range.append((2, range_top))
                    else:
                        shape_range.append((range_bottom, range_top))
                else:
                    shape_range.append((2, None))
            else:
                shape_range.append((dim_val, dim_val))
        ins_list[index]["range"] = tuple(shape_range)
    return ins_list


# 'pylint: disable=unused-argument,invalid-name,too-many-arguments,too-many-locals
def bn_training_update_v3_compute(x, sum, square_sum, scale, offset,
                                  y, batch_mean, batch_variance,
                                  reserve_1, reserve_2, epsilon,
                                  kernel_name="bn_training_update_v3"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: TVM tensor
        contains sum data
    square_sum: TVM tensor
        contains square_sum data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_1: dict
        dict of batch_mean, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_2: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_variance`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v3"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update_v3 compute
    """
    shape_x = list(x.shape)

    # runtime tiling: "NCHW" or "NC1HWC0" reduce [0, 2, 3]
    num_rec = operation.var("num_rec", dtype="float32")
    batch_var_scaler = operation.var("batch_var_scaler", dtype="float32")
    add_compile_info("bn_update_num_rec_dtype", "float32")
    add_compile_info("bn_update_batch_var_scaler_dtype", "float32")

    # compute the saved mean of x
    save_mean_reduce = tbe.vmuls(sum, num_rec)

    # compute the saved variance of x
    variance_div = tbe.vmuls(square_sum, num_rec)
    variance_square = tbe.vmul(save_mean_reduce, save_mean_reduce)
    save_variance_reduce = tbe.vsub(variance_div, variance_square)

    # compute the oefficient of y
    multiplier_add = tbe.vadds(save_variance_reduce, epsilon)
    multiplier_sqrt = tbe.vsqrt(multiplier_add)
    multiplier_div = tbe.vdiv(scale, multiplier_sqrt)
    multiplier = tbe.broadcast(multiplier_div, shape_x)

    addend_mul = tbe.vmul(multiplier_div, save_mean_reduce)
    addend_sub = tbe.vsub(offset, addend_mul)
    addend = tbe.broadcast(addend_sub, shape_x)

    # compute the batch normalization of x
    if x.dtype == "float16":
        x = tbe.cast_to(x, "float32")
        res_y = tbe.vadd(tbe.vmul(multiplier, x), addend)
        res_y = tbe.cast_to(res_y, "float16")
    else:
        res_y = tbe.vadd(tbe.vmul(multiplier, x), addend)

    # compute batch_mean and batch_var
    res_batch_mean = tbe.vmuls(sum, num_rec)
    res_batch_var = tbe.vmuls(save_variance_reduce, batch_var_scaler)

    res = [res_y, res_batch_mean, res_batch_var,
           save_mean_reduce, save_variance_reduce]

    return res


def bn_training_update_v3_fusion_compute(x, sum, square_sum, scale, offset,
                                         y, batch_mean, batch_variance,
                                         reserve_1, reserve_2, epsilon,
                                         kernel_name="bn_training_update_v3"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: TVM tensor
        contains sum data
    square_sum: TVM tensor
        contains square_sum data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_1: dict
        dict of batch_mean, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_2: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_variance`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v3"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update_v3 compute
    """
    fusion_util.check_fusion_input([x, sum])
    dict_x = fusion_util.extract_dict(x)
    dict_sum = fusion_util.extract_dict(sum)
    shape_x, shape_sum = fusion_util.normalize_shape([dict_x, dict_sum])

    in_x = fusion_util.create_placeholder(x, shape_x)
    in_sum = fusion_util.create_placeholder(sum, shape_sum)
    in_sqrsum = fusion_util.create_placeholder(square_sum, shape_sum)
    in_scale = fusion_util.create_placeholder(scale, shape_sum)
    in_offset = fusion_util.create_placeholder(offset, shape_sum)
    res = bn_training_update_v3_compute(in_x, in_sum, in_sqrsum,
                                        in_scale, in_offset,
                                        y, batch_mean, batch_variance,
                                        reserve_1, reserve_2,
                                        epsilon, kernel_name=kernel_name)

    return {"op_placeholder": [in_x, in_sum, in_sqrsum, in_scale, in_offset],
            "op_res": list(res)}


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn_training_update_v3(x, sum, square_sum, scale, offset,
                          y, batch_mean, batch_variance, reserve_1, reserve_2,
                          epsilon, kernel_name="bn_training_update_v3"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A 5HD Tensor for sum.
        The output of batch_normalization_forward_training_reduce.
    square_sum: dict
        dict of square_sum, A 5HD Tensor for square_sum.
        The output of batch_normalization_forward_training_reduce.
    scale: dict
        dict of scale, A 5HD Tensor for mean.
    offset: dict
        dict of offset, A 5HD Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_1: dict
        dict of batch_mean, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_2: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_variance`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v3"

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype").lower()
    dtype_sum = sum.get("dtype").lower()
    dtype_sqrsum = square_sum.get("dtype").lower()
    dtype_scale = scale.get("dtype").lower()
    dtype_offset = offset.get("dtype").lower()

    shape_x = x.get("shape")
    shape_sum = sum.get("shape")
    shape_sqrsum = square_sum.get("shape")
    shape_scale = scale.get("shape")
    shape_offset = offset.get("shape")

    data_format = x.get("format").upper()
    origin_format = x.get("ori_format").upper()

    # check dtype
    _check_dtype(dtype_x, dtype_sum, dtype_sqrsum,
                 dtype_scale, dtype_offset)

    # check format
    check_list = ("NC1HWC0", "NCHW")
    para_check.check_format(data_format, check_list, param_name="x")
    if data_format == "NCHW" and origin_format not in ("NCHW",):
        error_detail = "The origin format only supports NCHW when format is NCHW, origin_format:", origin_format
        error_manager_vector.raise_err_specific_reson("bn_training_update_v3", error_detail)

    x["shape"] = shape_x
    sum["shape"] = [1, shape_sum[1], 1, 1, shape_sum[4]]
    square_sum["shape"] = [1, shape_sqrsum[1], 1, 1, shape_sqrsum[4]]
    scale["shape"] = [1, shape_scale[1], 1, 1, shape_scale[4]]
    offset["shape"] = [1, shape_offset[1], 1, 1, shape_offset[4]]

    ins_list = [x, sum, square_sum, scale, offset]
    ins = classify(_refine_ins_list(ins_list), OpPatternMode.ELEWISE_WITH_BROADCAST,
                   extra_params={"disable_optimization": True})

    schedules = []
    tensors = []

    for (ins_x, ins_sum, ins_square_sum, ins_scale, ins_offset) in ins:
        with tbe.compute():
            shape_x, shape_sum, shape_sqrsum, shape_scale, shape_offset = \
                shape_util.variable_shape([ins_x, ins_sum, ins_square_sum, ins_scale, ins_offset])

            in_x = tvm.placeholder(shape_x, name="x", dtype=dtype_x)
            in_sum = tvm.placeholder(shape_sum, name="sum", dtype=dtype_sum)
            in_sqrsum = tvm.placeholder(shape_sum, name="sqrsum", dtype=dtype_sum)
            in_scale = tvm.placeholder(shape_sum, name="scale", dtype=dtype_sum)
            in_offset = tvm.placeholder(shape_sum, name="offset", dtype=dtype_sum)
            res = bn_training_update_v3_compute(in_x, in_sum, in_sqrsum,
                                                in_scale, in_offset,
                                                y, batch_mean, batch_variance,
                                                reserve_1, reserve_2,
                                                epsilon, kernel_name=kernel_name)
            tensors.append([in_x, in_sum, in_sqrsum, in_scale, in_offset] + res)

            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
