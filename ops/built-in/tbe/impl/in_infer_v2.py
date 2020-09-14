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

in_training_update
"""
from __future__ import division

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.utils.op_utils import *

NONETYPE = type(None)


# pylint: disable=locally-disabled,too-many-arguments,redefined-builtin
# pylint: disable=locally-disabled,invalid-name,too-many-locals,unused-argument


def _check_shape(shape, data_format="NC1HWC0", param_name="x"):
    """
     Function to check input tensors dims.

     Parameters
     ----------
     shape: list or tuple
         data shape of test input
     data_format: str
         format of input data
     Returns
     -------
     None
     """
    check_shape(shape, min_rank=5, max_rank=5,
                param_name=param_name)
    check_format(data_format.upper(), ("NC1HWC0",),
                 param_name=param_name)


def check_rule(data, rule_desc, param_name=PARAM_NAME):
    """
    The special check rule for tensor
    """
    if data is None or rule_desc is None:
        return
    error_info = {}
    error_info['errCode'] = OP_ERROR_CODE_009
    error_info['op_name'] = OP_NAME
    error_info['param_name'] = param_name
    error_info['rule_desc'] = rule_desc
    error_info['param_value'] = data
    raise RuntimeError(error_info,
                       "Op[%s] has rule: %s, but [%s] is [%s]."\
                        % (error_info['op_name'],
                        error_info['rule_desc'],
                        error_info['param_name'],
                        error_info['param_value']))


def _check_dims_equal(shape_x, shape):
    """
    Function to check the dimension C to be equal.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape: list or tuple
        data shape of test input
    data_format: str
        format of input data

    Returns
    -------
    None
    """
    if shape_x[0] != shape[0] or \
            shape_x[1] != shape[1] or shape_x[4] != shape[4]:
        check_rule("{} and {}".format(shape_x, shape),
                   "The dimensions N, C1, C0 of shape_x"\
                   "and shape must be equal",
                   "shape_x and shape")
    if shape[2] != 1 or shape[3] != 1:
        check_rule("{} and {}".format(shape[2], shape[3]),
                   "Dimension H,W must be 1",
                   "H,W")


@fusion_manager.register("in_training_update_v2")
def in_infer_compute(x,
                     gamma, beta, mean, variance,
                     y, mean_out, variance_out,
                     epsilon, kernel_name="in_training_update_v2"):
    """
    algorithm: instance_norm_v2
    instance normalization.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    gamma: TVM tensor
        contains scale data
    beta: TVM tensor
        contains offset data
    mean: TVM tensor
        contains mean data
    variance: TVM tensor
        contains variance data
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    mean_out: dict
        dict of mean, A `Tensor`. The update mean of save mean and running mean.
    variance_out: dict
        dict of variance, A `Tensor`.
        The update variance of save variance and running variance.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "in_training_update"

    Returns
    -------
    res: TVM tensor list
        the result of in_training_update compute
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)

    # compute the instance normalization of x
    is_cast = False
    if x.dtype == "float16":
        is_cast = True
        x = te.lang.cce.cast_to(x, "float32")
    mean_board = te.lang.cce.broadcast(mean, shape_x)

    # compute the saved variance of x
    var_board = te.lang.cce.broadcast(variance, shape_x)

    # (x - mean) / sqrt(var + eps)
    # x_mean = x - mean
    # multiplier_add = var + eps
    # multiplier_sqrt = sqrt(var + eps)

    x_mean = te.lang.cce.vsub(x, mean_board)
    multiplier_add = te.lang.cce.vadds(var_board, epsilon)
    multiplier_sqrt = te.lang.cce.vsqrt(multiplier_add)
    mean_wo_scale = te.lang.cce.vdiv(x_mean, multiplier_sqrt)

    result = mean_wo_scale
    if gamma is not None and beta is not None:
        gamma = te.lang.cce.broadcast(gamma, shape_x)
        beta = te.lang.cce.broadcast(beta, shape_x)
        gamma_scale = te.lang.cce.vmul(result, gamma)
        result = te.lang.cce.vadd(gamma_scale, beta)

    if is_cast:
        result = te.lang.cce.cast_to(result, "float16")
    scalar_zero = 0.0
    res_batch_mean = te.lang.cce.vadds(mean, scalar_zero)
    res_batch_var = te.lang.cce.vadds(variance, scalar_zero)

    res = [result, res_batch_mean, res_batch_var]
    return res


@check_op_params(REQUIRED_INPUT, OPTION_INPUT, OPTION_INPUT,
                 REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_OUTPUT, OPTION_OUTPUT, OPTION_OUTPUT,
                 OPTION_ATTR_FLOAT, KERNEL_NAME)
def in_infer_v2(x, gamma, beta, mean, variance,
                y, batch_mean, batch_variance,
                epsilon=0.00001, kernel_name="in_infer_v2"):
    """
    algorithm: instance_norm_infer
    instance normalization.

    Parameters
    ----------
    x: dict
        dict of input, A Tensor for input data.
    gamma: dict
        dict of scale, A Tensor for mean.
    beta: dict
        dict of offset, A Tensor for variance.
    mean: dict
        dict of mean, A Tensor for mean.
    variance: dict
        dict of variance, A Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of mean, A `Tensor`. The update mean of save mean and running mean.
    batch_variance: dict
        dict of variance, A `Tensor`.
        The update variance of save variance and running variance.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "in_infer_v2"

    Returns
    -------
    None
    """
    data_format = x.get("format")
    shape_x = x.get("shape")

    _check_shape(shape_x, data_format, "x")

    dtype_x = x.get("dtype")
    check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")

    x_input = tvm.placeholder(shape_x, name="x_input",
                              dtype=dtype_x.lower())
    gamma_input, beta_input, mean_input, var_input = None, None, None, None

    # Process gamma and beta
    has_gamma = False
    if gamma is not None and beta is not None:
        has_gamma = True

        shape_gamma = gamma.get("shape")
        dtype_gamma = gamma.get("dtype")
        _check_shape(shape_gamma, data_format, "gamma")
        _check_dims_equal(shape_x, shape_gamma)

        shape_beta = beta.get("shape")
        dtype_beta = beta.get("dtype")
        _check_shape(shape_beta, data_format, "beta")
        _check_dims_equal(shape_x, shape_beta)

        check_dtype(dtype_gamma.lower(), ("float16", "float32"),
                    param_name="gamma")
        check_dtype(dtype_beta.lower(), ("float16", "float32"),
                    param_name="beta")

        gamma_input = tvm.placeholder(shape_gamma, name="gamma_input",
                                      dtype=dtype_gamma.lower())
        beta_input = tvm.placeholder(shape_beta, name="beta_input",
                                     dtype=dtype_beta.lower())

    has_mean = False
    if mean is not None and variance is not None:
        has_mean = True
        # Process mean and var
        shape_mean = mean.get("shape")
        dtype_mean = mean.get("dtype")
        _check_shape(shape_mean, data_format, "mean")
        _check_dims_equal(shape_x, shape_mean)

        shape_var = variance.get("shape")
        dtype_var = variance.get("dtype")
        _check_shape(shape_var, data_format, "variance")
        _check_dims_equal(shape_x, shape_var)

        check_dtype(dtype_mean.lower(), ("float16", "float32"),
                    param_name="mean")
        check_dtype(dtype_var.lower(), ("float16", "float32"),
                    param_name="variance")

        mean_input = tvm.placeholder(shape_mean, name="mean_input",
                                     dtype=dtype_mean.lower())
        var_input = tvm.placeholder(shape_var, name="variance_input",
                                    dtype=dtype_var.lower())

    res = in_infer_compute(x_input,
                           gamma_input, beta_input, mean_input, var_input,
                           y, batch_mean, batch_variance, epsilon,
                           kernel_name=kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    if has_gamma:
        tensor_list = [x_input, gamma_input, beta_input, mean_input,
                       var_input] + list(res)
    else:
        tensor_list = [x_input, mean_input,
                       var_input] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
