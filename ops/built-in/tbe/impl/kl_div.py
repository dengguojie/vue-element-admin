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

kl_div
"""

from functools import reduce as reduce_one_dim
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util

SHAPE_SIZE_LIMIT = 2**30  # shape limit

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("kl_div")
def kl_div_compute(input_x,
                   input_target,
                   output_y,
                   reduction,
                   batch_size,
                   kernel_name="kl_div"):
    """
    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    input_target : TVM tensor
        the placeholder of input_target
    output_y : dict
        shape and dtype of output, should be same shape and type as input_x
    reduction: str
        Specifies the reduction to apply to the output:
        reduction="batchmean" or reduction="sum".
        "batchmean": the sum of the output will be divided by the batchsize
        "sum": the output will be summed
    batch_size: int
        Equal to the first dimension value of the input shape.
    kernel_name : str
        cce kernel name, default value is "kl_div"

    Returns
    ------
    compute result of kl_div
    """
    log_target = te.lang.cce.vlog(input_target, 1)
    tmp_result = te.lang.cce.vsub(log_target, input_x)
    output_pos = te.lang.cce.vmul(input_target, tmp_result)

    # max(output_pos, 0)
    target_gt_zero = te.lang.cce.vmaxs(input_target, 0)

    add_espmin = te.lang.cce.vadds(target_gt_zero, 1.18e-38)
    y_espmin = te.lang.cce.vdiv(target_gt_zero, add_espmin)
    output_res = te.lang.cce.vmul(y_espmin, output_pos)

    if reduction == "batchmean":
        output_res = te.lang.cce.vmuls(output_res, 1.0 / batch_size)
        final_res = te.lang.cce.sum(output_res, axis=0)
    elif reduction == "sum":
        final_res = te.lang.cce.sum(output_res, axis=0)
    else:
        raise RuntimeError(
            "Reduction method only support batchmean and sum")

    return final_res

def _check_parameter(input_x, input_target):
    """
    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    input_target : dict
        shape and dtype of input_target.Shape and dtype must be same as input_x
    Returns
    ------
    None
    """
    shape_x = input_x.get("shape")
    shape_target = input_target.get("shape")
    if list(shape_x) != list(shape_target):
        raise RuntimeError("input_x and input_target must "
                           "have the same shape.")
    util.check_shape_rule(shape_x)
    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)

    # check input tensor data_type and kernel_name
    check_list = ("float32")
    dtype_x = input_x.get("dtype").lower()
    dtype_target = input_target.get("dtype").lower()
    if dtype_x != dtype_target:
        raise RuntimeError("input_x and input_target must "
                           "have the same dtype.")
    util.check_dtype_rule(dtype_x, check_list)

    if dtype_x == "float32" and not tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vmul", "float32"):
        raise RuntimeError(
            "Instric only support float16 while input dtype is float32")


@util.check_input_type(dict, dict, dict, str, str)
def kl_div(input_x, input_target, output_y, reduction, kernel_name="kl_div"):
    """
    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    input_target : dict
        shape and dtype of input_target.Shape and dtype must be same as input_x
    output_y : dict
        shape and dtype of output.Shape and dtype must be same as input_x
    reduction: str
        Specifies the reduction to apply to the output:
        reduction="batchmean" or reduction="sum".
        "batchmean": the sum of the output will be divided by the batchsize
        "sum": the output will be summed
    kernel_name : str
        cce kernel name, default value is "kl_div"

    Returns
    ------
    None
    """
    # check input parameter
    _check_parameter(input_x, input_target)

    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    batch_size = shape_x[0]
    util.check_kernel_name(kernel_name)
    shape_one_dim = [reduce_one_dim(lambda x, y: x * y, shape_x[:])]
    data_x = tvm.placeholder(shape_one_dim,
                             name="data_x",
                             dtype=dtype_x)
    data_target = tvm.placeholder(shape_one_dim,
                                  name="data_target",
                                  dtype=dtype_x)

    final_res = kl_div_compute(data_x,
                               data_target,
                               output_y,
                               reduction,
                               batch_size,
                               kernel_name=kernel_name)
    with tvm.target.cce():
        auto_sch = generic.auto_schedule(final_res)

    config = {
        "name": kernel_name,
        "tensor_list": (data_x, data_target, final_res)
    }

    te.lang.cce.cce_build_code(auto_sch, config)
