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

batch_norm
"""

from __future__ import absolute_import
from __future__ import division
from te import platform as tbe_platform
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import *
from topi import generic
from topi.cce import util


NONETYPE = type(None)
CONST_NEWTON_FACTOR1 = 3.0
CONST_ONE = 1
CONST_HALF = 0.5
CONST_NEG_ONE = -1.0


# pylint: disable=locally-disabled,too-few-public-methods,no-init
def _format_check(arg_input):
    """
    Function to check if the data_format is in line with norms.

    Parameters
    ----------
    input: dict
        dict of input
    data_format: str
        format of input data

    Returns

    -------
    None
    """
    format_data = arg_input.get("format")
    excepted_format_list = ["ND", "NC1HWC0", "NCHW", "NHWC"]
    check_format(format_data, excepted_format_list)


def _check_dims_equal(shape_x, shape, data_format):
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

    if data_format in ("ND", "NCHW", "NHWC"):

        if len(shape_x) == 1:
            index_c = 0
        elif data_format != "NHWC":
            index_c = 1
        else:
            index_c = 3
        if shape_x[index_c] != shape[0]:
            raise RuntimeError(
                "Dimensions must be equal")


def _check_shape_dims(shape, data_format):
    """
    Function to check input tensors must be 5D ones.

    Parameters
    ----------
    shape: list or tuple
        data shape of test input
    data_format: str
        format of input data
    is_x: bool
        data to check is input_x or not

    Returns
    -------
    None
    """
    if data_format == "NC1HWC0":
        if len(shape) != 5:
            raise RuntimeError(
                "shape is invalid, which only support 5D Tensor")


def param_scale_check(shape_x, shape_scale):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x : list or tuple.
        shape of x.
    shape_scale : list or tuple.
        shape of scale.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if not(length_scale == 1 and shape_scale[0] == 1):
        if length_x != length_scale:
            error_info = {}
            error_info['errCode'] = 'E81014'
            error_info['real_x_dims'] = str(length_x)
            error_info['real_scale_dims'] = str(length_scale)
            raise RuntimeError(error_info,
                "In op[scale], the dims of input tensor x and tensor scale should be equal, but actually are [%s] and [%s]."
                %(error_info['real_x_dims'], error_info['real_scale_dims']))

        for i in range(length_scale):
            if shape_scale[i] != shape_x[i] and shape_scale[i] != 1:
                error_info = {}
                error_info['errCode'] = 'E80013'
                error_info['opname'] = 'scale'
                error_info['input1_name'] = 'x'
                error_info['input2_name'] = 'scale'
                error_info['input1_shape'] = str(shape_x)
                error_info['input2_shape'] = str(shape_scale)
                raise RuntimeError(error_info,
                    "In op[%s], the inputs[%s][%s] could not be broadcast together with shapes[%s][%s]."
                    %(error_info['opname'], error_info['input1_name'], error_info['input2_name'],
                      error_info['input1_shape'], error_info['input2_shape']))


# pylint: disable=locally-disabled,too-many-arguments
def _shape_check(shape_x, shape_mean, shape_variance, scale, format_x):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_scale: list or tuple
        shape_scale's data shape
    shape_offset: list or tuple
        shape_offset's data shape
    shape_mean: list or tuple
        shape_mean's data shape
    shape_variance: list or tuple
        shape_variance's data shape
    is_training: bool
        A bool value to indicate the operation is for training or inference.

    Returns
    -------
    None
    """

    check_shape(shape_x)
    if format_x in ["NHWC", "NCHW", "ND"]:
        check_shape(shape_mean, max_rank=1)
        check_shape(shape_variance, max_rank=1)
    _check_shape_dims(shape_x, format_x)

    _check_dims_equal(shape_x, shape_mean, format_x)
    _check_dims_equal(shape_x, shape_variance, format_x)

    if scale is not None:
        shape_scale = scale.get("shape")
        param_scale_check(shape_x, shape_scale)


# pylint: disable=invalid-name,redefined-outer-name
def _fused_scale_bias_compute(x, mean, variance, scale, bias):
    """
    algorithm: Scale
    y = scale*x

    Parameters
    ----------
    x : TVM tensor
        contains x data
    scale : TVM tensor
        contains scale data

    Returns
    -------
    res: TVM tensor list
        the result of scale compute
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    dtype_x = x.dtype
    dtype_scale = scale.dtype
    dtype_bias = bias.dtype
    mean_broadcast = te.lang.cce.broadcast(mean, shape_x)
    var_broadcast = te.lang.cce.broadcast(variance, shape_x)
    mean_add = te.lang.cce.vadd(x, mean_broadcast)
    res_y = te.lang.cce.vmul(var_broadcast, mean_add)

    is_cast = False
    product_version = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if product_version not in ("Ascend310", "Hi3796CV300ES"):
        if dtype_x == "float16":
            is_cast = True
            res_y = te.lang.cce.cast_to(res_y, 'float32')
        if dtype_scale == "float16":
            scale = te.lang.cce.cast_to(scale, 'float32')
        if dtype_bias == "float16":
            bias = te.lang.cce.cast_to(bias, 'float32')

    scale_broad = te.lang.cce.broadcast(scale, shape_x)
    bias_broad = te.lang.cce.broadcast(bias, shape_x)

    res_tmp = te.lang.cce.vmul(res_y, scale_broad)
    res = te.lang.cce.vadd(res_tmp, bias_broad)

    if is_cast:
        res = te.lang.cce.cast_to(res, dtype_x)
    return res


# pylint: disable=invalid-name,redefined-outer-name
def _fused_scale_compute(x, mean, variance, scale):
    """
    algorithm: Scale
    y = scale*x

    Parameters
    ----------
    x : TVM tensor
        contains x data
    scale : TVM tensor
        contains scale data

    Returns
    -------
    res: TVM tensor list
        the result of scale compute
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    dtype_x = x.dtype
    dtype_scale = scale.dtype

    mean_broadcast = te.lang.cce.broadcast(mean, shape_x)
    var_broadcast = te.lang.cce.broadcast(variance, shape_x)
    mean_add = te.lang.cce.vadd(x, mean_broadcast)
    res_y = te.lang.cce.vmul(var_broadcast, mean_add)

    is_cast = False
    product_version = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    if product_version not in ("Ascend310", "Hi3796CV300ES"):
        if dtype_x == "float16":
            is_cast = True
            res_y = te.lang.cce.cast_to(res_y, 'float32')
        if dtype_scale == "float16":
            scale = te.lang.cce.cast_to(scale, 'float32')

    scale_broad = te.lang.cce.broadcast(scale, shape_x)

    res = te.lang.cce.vmul(res_y, scale_broad)

    if is_cast:
        res = te.lang.cce.cast_to(res, dtype_x)
    return res


# pylint: disable=invalid-name,redefined-outer-name
def _fused_compute(x, mean, variance):
    """
    algorithm: Scale
    y = scale*x

    Parameters
    ----------
    x : TVM tensor
        contains x data
    scale : TVM tensor
        contains scale data

    Returns
    -------
    res: TVM tensor list
        the result of scale compute
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    mean_broadcast = te.lang.cce.broadcast(mean, shape_x)
    var_broadcast = te.lang.cce.broadcast(variance, shape_x)
    mean_add = te.lang.cce.vadd(x, mean_broadcast)
    res_y = te.lang.cce.vmul(var_broadcast, mean_add)
    return res_y


# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name,protected-access
@fusion_manager.register("bninference_d")
def bninference_d_compute(x, mean, variance, scale, bias, y,
                          momentum, epsilon, use_global_stats, mode):
    """
    Parameters
    ----------
    x: dict
        contains x data. A 4D or 5D Tensor of type float16 or float32.
    mean: dict
        contains mean data.Must be 1D if input "x" Specifies the mean used for inference.
    variance: dict
        contains variance data.Must be 1D if input "x" Specifies the variance used for inference.
    scale: dict
        no use in caffe batchnorm inference
    bias: dict
        no use in caffe batchnorm inference
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    momentum: float
        a float number of the variance and mean's scale factor
    epsilon: float
        a small float number added to the variance of x to avoid dividing by zero. Defaults to "0.00001".
    use_global_stats: bool
        means the caffe inference model, only can be True.
    mode: int
        an optional attr, no use
    kernel_name: str
        kernel name

    Returns
    -------
    res: TVM tensor list
        the result of batch_norm_ext2 compute
    """
    if scale is not None and bias is not None:
        res = _fused_scale_bias_compute(x, mean, variance, scale, bias)
    elif scale is not None and bias is None:
        res = _fused_scale_compute(x, mean, variance, scale)
    else:
        res = _fused_compute(x, mean, variance)

    return res


def _dtype_scale_offset_check(x, mean, variance, scale, offect):
    dtype_x = x.get("dtype")
    dtype_mean = mean.get("dtype")
    dtype_variance = variance.get("dtype")
    product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    if product in ("Hi3796CV300ES"):
        checklist = ["float16"]
    else:
        checklist = ["float32", "float16"]
    check_dtype(dtype_mean.lower(), checklist)
    check_dtype(dtype_x.lower(), checklist)
    check_dtype(dtype_variance.lower(), checklist)

    if scale is not None:
        dtype_scale = scale.get("dtype")
        check_dtype(dtype_scale.lower(), checklist)
    if offect is not None and bool(offect):
        dtype_offect = offect.get("dtype")
        check_dtype(dtype_offect.lower(), checklist)


def _dtype_check(x, mean, variance):
    dtype_x = x.get("dtype")
    dtype_mean = mean.get("dtype")
    dtype_variance = variance.get("dtype")
    product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    if product in ("Hi3796CV300ES"):
        checklist = ["float16"]
    else:
        checklist = ["float32", "float16"]
    check_dtype(dtype_mean.lower(), checklist)
    check_dtype(dtype_x.lower(), checklist)
    check_dtype(dtype_variance.lower(), checklist)


def para_shape_scale_offset_check(x, mean, variance, scale, offect, format_x):
    """
    :param x:input tensor
    :param mean:mean tensor
    :param variance: var tensor
    :param format_x: format tensor
    :return: None
    """
    shape_mean = mean.get("shape")
    shape_variance = variance.get("shape")
    shape_x = x.get("shape")
    check_shape(shape_mean)
    check_shape(shape_variance)

    if scale is not None:
        shape_scale = scale.get("shape")
        check_shape(shape_scale)
    if offect is not None and bool(offect):
        shape_offect = offect.get("shape")
        check_shape(shape_offect)

    _shape_check(shape_x, shape_mean, shape_variance, scale, format_x)


def para_shape_check(x, mean, variance, scale, format_x):
    """
    :param x:input tensor
    :param mean:mean tensor
    :param variance: var tensor
    :param format_x: format tensor
    :return: None
    """
    shape_mean = mean.get("shape")
    shape_variance = variance.get("shape")
    shape_x = x.get("shape")
    check_shape(shape_mean)
    check_shape(shape_variance)
    _shape_check(shape_x, shape_mean, shape_variance, scale, format_x)

def para_scale_bias_check(x, mean, variance, scale, offect, use_global_stats, kernel_name):
    """
    :param x:input tensor
    :param mean: mean tensor
    :param variance: var tensor
    :param use_global_stats: inference type
    :param kernel_name: kernel_name
    :return: none
    """
    format_x = x.get("format")
    _format_check(x)
    _dtype_scale_offset_check(x, mean, variance, scale, offect)
    if not use_global_stats:
        dictArgs = {}
        dictArgs['errCode'] = 'E80000'
        dictArgs['opname'] = 'batchnorm'
        dictArgs['param_name'] = 'use_global_stats'
        dictArgs['excepted_value'] = 'True'
        dictArgs['real_value'] = str(use_global_stats)
        raise RuntimeError(dictArgs,
                           "In op[%s], the parameter[%s] should be [%s], but actually is [%s]."
                           % (dictArgs['opname'], dictArgs['param_name'],
                              dictArgs['excepted_value'],
                              dictArgs['real_value']))
    para_shape_scale_offset_check(x, mean, variance, scale, offect, format_x)

def para_check(x, mean, variance, scale, use_global_stats, kernel_name):
    """
    :param x:input tensor
    :param mean: mean tensor
    :param variance: var tensor
    :param use_global_stats: inference type
    :param kernel_name: kernel_name
    :return: none
    """
    format_x = x.get("format")
    _format_check(x)
    _dtype_check(x, mean, variance)
    if not use_global_stats:
        dictArgs = {}
        dictArgs['errCode'] = 'E80000'
        dictArgs['opname'] = 'batchnorm'
        dictArgs['param_name'] = 'use_global_stats'
        dictArgs['excepted_value'] = 'True'
        dictArgs['real_value'] = str(use_global_stats)
        raise RuntimeError(dictArgs,
                           "In op[%s], the parameter[%s] should be [%s], but actually is [%s]."
                           % (dictArgs['opname'], dictArgs['param_name'],
                              dictArgs['excepted_value'],
                              dictArgs['real_value']))
    para_shape_check(x, mean, variance, scale, format_x)


def get_param_scale_shape(shape_x, shape_scale):
    """
    Function to calculate the shape of scale.

    Parameters
    ----------
    shape_x : list or tuple.
        shape of x.
    shape_scale : list or tuple.
        shape of scale.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if length_scale == 1 and shape_scale[0] == 1:
        shape = [1] * length_x
    else:
        shape = list(shape_scale)

    return shape


def gen_tensor(x, mean, variance, scale, offect):
    """
    :param x:x tensor
    :param mean: mean tensor
    :param variance:var tensor
    :return:
    x_input:x
    mean_input:mean
    var_input:var
    scale:scale,not use
    b:not use
    """
    shape_x = x.get("shape")
    format_x = x.get("format")
    dtype_x = x.get("dtype")
    if format_x in ("ND", "NCHW"):
        if len(shape_x) == 1:
            index_c = 0
        else:
            index_c = 1
    elif format_x == "NHWC":
        if len(shape_x) == 1:
            index_c = 0
        else:
            index_c = 3
    else:
        c1 = shape_x[1]
        c0 = shape_x[4]
    shape_mean = mean.get("shape")
    shape_variance = variance.get("shape")

    if format_x in ("ND", "NCHW", "NHWC"):
        shape_mean = [1] * len(shape_x[:index_c]) + list(shape_mean) \
                     + [1] * len(shape_x[index_c + 1:])
        shape_variance = [1] * len(shape_x[:index_c]) + list(shape_variance) \
                         + [1] * len(shape_x[index_c + 1:])
    else:
        shape_mean = [1] + [c1] + [1, 1] + [c0]
        shape_variance = [1] + [c1] + [1, 1] + [c0]

    shape_scale = {}
    shape_offect = {}
    if scale is not None:
        shape_scale = scale.get("shape")
    if offect is not None and bool(offect):
        shape_offect = offect.get("shape")

    x_input = tvm.placeholder(shape_x, name="x", dtype=dtype_x.lower())
    mean_input = tvm.placeholder(shape_mean, name="mean",
                                 dtype=dtype_x.lower())
    variance_input = tvm.placeholder(shape_variance, name="variance",
                                     dtype=dtype_x.lower())

    scale_input = None
    offect_input = None
    if len(shape_scale) > 0:
        dtype_scale = scale.get("dtype")
        shape_scale_new = get_param_scale_shape(shape_x, shape_scale)
        scale_input = tvm.placeholder(shape_scale_new, name="scale_input",
                                      dtype=dtype_scale.lower())
        if len(shape_offect) > 0:
            dtype_offect = offect.get("dtype")
            shape_offect_new = shape_scale_new
            offect_input = tvm.placeholder(shape_offect_new, name="offect_input",
                                           dtype=dtype_offect.lower())

    return x_input, mean_input, variance_input, scale_input, offect_input


# pylint: disable=locally-disabled,no-member
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, OPTION_INPUT,
                 OPTION_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_FLOAT, REQUIRED_ATTR_FLOAT,
                 REQUIRED_ATTR_BOOL, REQUIRED_ATTR_INT, KERNEL_NAME)
def bninference_d(x, mean, variance, scale, offect, y, momentum, epsilon,
                  use_global_stats, mode, kernel_name="bninference"):
    """
    
    Parameters
    ----------
    x: dict
        contains x data. A 4D or 5D Tensor of type float16 or float32.
    mean: dict
        contains mean data.Must be 1D if input "x" Specifies the mean used for inference.
    variance: dict
        contains variance data.Must be 1D if input "x" Specifies the variance used for inference.
    scale: dict
        no use in caffe batchnorm inference
    bias: dict
        no use in caffe batchnorm inference
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    momentum: float
        a float number of the variance and mean's scale factor
    epsilon: float
        a small float number added to the variance of x to avoid dividing by zero. Defaults to "0.00001".
    use_global_stats: bool
        means the caffe inference model, only can be True.
    mode: int
        an optional attr, no use
    kernel_name: str
        kernel name

    Returns
    -------
    None
    """
    if offect is not None or scale is not None:
        para_scale_bias_check(x, mean, variance, scale, offect, use_global_stats, kernel_name)
    else:
        para_check(x, mean, variance, scale, use_global_stats, kernel_name)
    x_input, mean_input, variance_input, scale_input, offect_input = gen_tensor(x, mean, variance, scale, offect)
    res = bninference_d_compute(x_input, mean_input,
                                variance_input, scale_input, offect_input,
                                y, momentum, epsilon,
                                use_global_stats, mode)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    if offect is None and scale is None:
        tensor_list = [x_input, mean_input, variance_input, res]
    elif offect is None and scale is not None:
        tensor_list = [x_input, mean_input, variance_input, scale_input, res]
    else:
        tensor_list = [x_input, mean_input, variance_input, scale_input, offect_input, res]
    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
