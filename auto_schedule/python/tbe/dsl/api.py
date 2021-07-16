#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
tbe dsl API:
In order to simplify the procedure of writing schedule, TBE provides a set of TensorEngine APIs.
Using those API to develop operators, you can use the "Auto_schedule" create schedule.
"""
from typing import Any
from typing import Dict
from typing import Optional

from .compute import cast
from .compute import conv2d_backprop_filter_compute as conv2d_dw_compute
from .compute import conv2d_backprop_input_compute as conv2d_dx_compute
from .compute import conv3d_backprop_filter_compute as conv3d_dw_compute
from .compute import conv3d_backprop_input_compute as conv3d_dx_compute
from .compute import conv3d_compute
from .compute import depthwise_conv2d_compute
from .compute import dilation_compute
from .compute import gemm_compute
from .compute import mmad_compute
from .compute import math
from .compute import nn
from .compute import reduce
from .compute import array
from .compute import inplace
from .compute import pooling2d as pooling2d_compute
from .compute import pooling3d as pooling3d_compute
from .compute import pooling3d_max_grad_grad as pooling3d_max_grad_grad_compute
from .unify_schedule import auto_schedule as tbe_auto_schedule
from .unify_schedule.build import build as tbe_build
from .base import shape_classifier
from .base import operation
import traceback


def ceil(raw_tensor, dtype="int32"):
    """
    cast tensor from src_type to dst_dtype with ceiling method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    dtype : string
        dst dtype need to cast to
    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.ceil(raw_tensor, dtype)


def floor(raw_tensor, dtype="int32"):
    """
    cast tensor from src_type to dst_dtype with flooring method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    dtype : string
        dst dtype need to cast to
    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.floor(raw_tensor, dtype)


def round(raw_tensor, dtype="int32"):
    """
    cast tensor from src_type to dst_dtype with rounding method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    dtype : string
        dst dtype need to cast to
    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.round(raw_tensor, dtype)


def trunc(raw_tensor, dtype="int32"):
    """
    cast tensor from src_type to dst_dtype with trunc method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    dtype : string
        dst dtype need to cast to
    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.trunc(raw_tensor, dtype)


def round_half_up(raw_tensor, dtype="int32"):
    """
    cast tensor from src_type to dst_dtype with rounding method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    dtype : string
        dst dtype need to cast to
    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.round_half_up(raw_tensor, dtype)


def cast_to(data, dtype, f1628IntegerFlag=True):
    """
    a wrapped cast operations , cast data to the type of dtype

    Parameters
    ----------
    data : tvm.tensor
        tensors need to change dtype

    dtype : string
        dst dtype need to cast to

    f1628IntegerFlag : bool
        before fp16->int8/uint8, the data is all interger or not. default value
        is False.

    Returns
    -------
    tensor : tvm.tensor
    """
    return cast.cast_to(data, dtype, f1628IntegerFlag)


def vadd(lhs, rhs):
    """
    calculate elewise add

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : lhs + rhs
    """
    return math.vadd(lhs, rhs)


def vsub(lhs, rhs):
    """
    calculate elewise sub

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : lhs - rhs
    """
    return math.vsub(lhs, rhs)


def vmul(lhs, rhs):
    """
    calculate elewise multiply

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        right hand tensor

    Returns
    -------
    wrapped_tensor : lhs*rhs
    """
    return math.vmul(lhs, rhs)


def vdiv(lhs, rhs):
    """
    calculate elewise div

    Parameters
    -----
    lhs: wrapped_tensor or tvm.tensor
         divisor tensor
    rhs: wrapped_tensor or tvm.tensor
         divided tensor

    returns
    -----
    wrapped_tensor: lhs / rhs
    """
    return math.vdiv(lhs, rhs)


def vrec(raw_tensor, impl_mode="high_performance"):
    """
    calculate vrec(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    impl_mode : only support high_performance and high_precision

    Returns
    -------
    wrapped_tensor : vrec(raw_tensor)
    """
    return math.vrec(raw_tensor, impl_mode)


def vmod(lhs, rhs):
    """
    calculate element-wise remainder of division

    Parameters
    -----
    lhs : wrapped_tensor or tvm.tensor
          left hand tensor

    rhs : wrapped_tensor or tvm.tensor
          right hand tensor

    Returns
    -----
    wrapped_tensor : lhs - floor(lhs/rhs) * rhs
    """
    return math.vmod(lhs, rhs)


def vmax(lhs, rhs):
    """
    calculate elewise compare, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    return math.vmax(lhs, rhs)


def vmin(lhs, rhs):
    """
    calculate elewise compare, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : min(lhs , rhs)
    """
    return math.vmin(lhs, rhs)


def vlog(raw_tensor, impl_mode="high_performance"):
    """
    calculate ln(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    impl_mode : only support high_performance and high_precision

    Returns
    -------
    wrapped_tensor : log(raw_tensor)
    """
    return math.vlog(raw_tensor, impl_mode)


def vexp(raw_tensor):
    """
    calculate exp(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : exp(raw_tensor)
    """
    return math.vexp(raw_tensor)


def vabs(raw_tensor):
    """
    calculate abs(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : abs(raw_tensor)
    """
    return math.vabs(raw_tensor)


def vsqrt(raw_tensor, impl_mode="high_performance"):
    """
    calculate vsqrt(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    impl_mode : only support high_performance and high_precision

    Returns
    -------
    wrapped_tensor : vsqrt(raw_tensor)
    """
    return math.vsqrt(raw_tensor, impl_mode)


def vrsqrt(raw_tensor, impl_mode="high_performance"):
    """
    calculate vrsqrt(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    impl_mode : only support high_performance and high_precision

    Returns
    -------
    wrapped_tensor : vrsqrt(raw_tensor)
    """
    return math.vrsqrt(raw_tensor, impl_mode)


def vnot(raw_tensor):
    """
    calculate vnot(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vnot(raw_tensor)
    """
    return math.vnot(raw_tensor)


def vor(lhs, rhs):
    """
    calculate bitwise or op, return the or value
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : or(lhs , rhs)
    """
    return math.vor(lhs, rhs)


def vand(lhs, rhs):
    """
    calculate bitwise and op, return the and value
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    return math.vand(lhs, rhs)


def vlogic(lhs, rhs=None, operation='logic_and'):
    """
    calculate elewise logic operation

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        right hand tensor

    operation : operator type, logic_and, logic_or, logic_not

    Returns
    -------
    wrapped_tensor
    """
    return math.vlogic(lhs, rhs, operation)


def vadds(raw_tensor, scalar):
    """
    add a tensor by a scalar, dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : raw_tensor + scalar
    """
    return math.vadds(raw_tensor, scalar)


def vmuls(raw_tensor, scalar):
    """
    multiply a tensor by a scalar, dtype of raw_tensor
    and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : raw_tensor*scalar
    """
    return math.vmuls(raw_tensor, scalar)


def vmaxs(raw_tensor, scalar):
    """
    Calculate elewise compare, return the max one of scalar or tensor's element,
    dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : max(raw_tensor, scalar)
    """
    return math.vmaxs(raw_tensor, scalar)


def vmins(raw_tensor, scalar):
    """
    Calculate elewise compare, return the min one of scalar or tensor's element,
     dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : min(raw_tensor, scalar)
    """
    return math.vmins(raw_tensor, scalar)


def vaxpy(lhs, rhs, scalar):
    """
    calculate elewise scalar*lhs + rhs, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    return math.vaxpy(lhs, rhs, scalar)


def vmla(tensor_0, tensor_1, tensor_2):
    """
    calculate x*tensor_1 + tensor_2,  only support float16, float32
    Parameters
    ----------
    x : wrapped_tensor or tvm.tensor
    tensor_1 : wrapped_tensor or tvm.tensor
    tensor_2 : wrapped_tensor or tvm.tensor
    Returns
    -------
    wrapped_tensor : X*tensor_1 + tensor_2
    """
    return math.vmla(tensor_0, tensor_1, tensor_2)


def vmadd(tensor_0, tensor_1, tensor_2):
    """
    calculate tensor_0*tensor_2 + tensor_1,  only support  float16, float32
    Parameters
    ----------
    tensor_0 : wrapped_tensor or tvm.tensor
    tensor_1 : wrapped_tensor or tvm.tensor
    tensor_2 : wrapped_tensor or tvm.tensor
    Returns
    -------
    wrapped_tensor : tensor_0*tensor_2 + tensor_1
    """
    return math.vmadd(tensor_0, tensor_1, tensor_2)


def vcmp(lhs, rhs, operation='lt', mode='bool'):
    """
    calculate elewise compare

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        right hand tensor

    operation : operator type, eq, ne, lt, gt, ge, le

    mode : bool, the dtype of return value is bool
           bit, the dtype of return value is uint8

    Returns
    -------
    wrapped_tensor
    """
    return math.vcmp(lhs, rhs, operation, mode)


def vsel(condition, lhs, rhs):
    """
    if condition = ture, the result is lhs,
        select

    Parameters
    ----------
    condition : wrapped_tensor or tvm.tensor, the dtype is bool or uint8

    lhs : wrapped_tensor or tvm.tensor or scalar

    rhs : wrapped_tensor or tvm.tensor or scalar

    Returns
    -------
    wrapped_tensor :
    """
    return math.vsel(condition, lhs, rhs)


def vcmpsel(lhs, rhs=None, operation='lt', slhs=None, srhs=None):
    """
    calculate elewise compare

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        compare left hand tensor
    rhs : wrapped_tensor or tvm.tensor or scalar
        compare right hand tensor or scalar
    operation : operator type, eq, ne, lt, gt, ge, le
    slhs : wrapped_tensor or tvm.tensor or scalar
        select left hand tensor or scalar
    srhs : wrapped_tensor or tvm.tensor or scalar
        select right hand tensor or scalar

    Returns
    -------
    wrapped_tensor
    """
    return math.vcmpsel(lhs, rhs, operation, slhs, srhs)


def vmaddrelu(tensor_0, tensor_1, tensor_2):
    """
    calculate relu(tensor_0*tensor_2 + tensor_1), only support  float16, float32
    Parameters
    ----------
    tensor_0 : wrapped_tensor or tvm.tensor
    tensor_1 : wrapped_tensor or tvm.tensor
    tensor_2 : wrapped_tensor or tvm.tensor
    Returns
    -------
    wrapped_tensor : relu(tensor_0*tensor_2 + tensor_1)
    """
    return nn.vmaddrelu(tensor_0, tensor_1, tensor_2)


def vaddrelu(lhs, rhs):
    """
    calculate relu(lhs + rhs)

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : relu (lhs + rhs)
    """
    return nn.vaddrelu(lhs, rhs)


def vsubrelu(lhs, rhs):
    """
    calculate relu(lhs - rhs)

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : relu (lhs - rhs)
    """
    return nn.vsubrelu(lhs, rhs)


def vrelu(raw_tensor):
    """
    calculate vrelu(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vrelu(raw_tensor)
    """
    return nn.vrelu(raw_tensor)


def vlrelu(raw_tensor, alpha=0):
    """
    calculate leaky_relu

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vlrelu(raw_tensor)
    """
    return nn.vlrelu(raw_tensor, alpha)


def clip(data, max_value, min_value):
    """
    round data to [min_value,max_value]

    Parameters
    ----------
    data : tvm.tensor
        tensors need to change dtype

    max_value/min_value : float
        the range of res

    Returns
    -------
    tensor : tvm.tensor ,elements in tensor is in range [min_value,max_value]
    """
    return nn.clip(data, max_value, min_value)


def broadcast(var, shape, output_dtype=None):
    """
    broadcast scalar to tensor, only support float16

    Parameters
    ----------
    var : can be python instance of int and float, or tvm.const

    shape : tensor shape

    output_dtype : tensor dtype , default : var.dtype

    Returns
    -------
    wrapped_tensor : broadcast tensor
    """
    return nn.broadcast(var, shape, output_dtype)


def reduce_sum(raw_tensor, axis, keepdims=False):
    """
    calculate reduce_sum of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    axis : int or list
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    Returns
    -------
    res : wrapped_tensor
    """
    return reduce.reduce_sum(raw_tensor, axis, keepdims)


def reduce_min(raw_tensor, axis, keepdims=False, impl_mode="high_performance"):
    """
    calculate reduce_min of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    axis : int or list
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    Returns
    -------
    res : wrapped_tensor
    """
    return reduce.reduce_min(raw_tensor, axis, keepdims, impl_mode)


def reduce_max(raw_tensor, axis, keepdims=False, impl_mode="high_performance"):
    """
    calculate reduce_max of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    axis : int or list
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    priority_flag : supported 1(precision) and 0(performance)
    Returns
    -------
    res : wrapped_tensor
    """
    return reduce.reduce_max(raw_tensor, axis, keepdims, impl_mode)


def reduce_prod(raw_tensor, axis, keepdims=False):
    """
    calculate reduce_prod of raw_tensor, only support float16
    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    axis : int
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    Returns
    -------
    res : wrapped_tensor
    """
    return reduce.reduce_prod(raw_tensor, axis, keepdims)


def split(data, split_dim, size_splits):
    """Split a tensor into len(size_splits) tensors along one dimension.

    Parameters
    ----------
    data: TVM tensor
        input tensor.
    split_dim: int
        the dimension along which to split.
    size_splits: list or tuple
        a Python list containing the sizes of each output tensor along `split_dim`.

    Returns
    -------
    output_shape_list: list
        the list of output shapes.
    output_tensor_list: list
        the list of output tensors, output tensor type is TVM tensor.
    """
    return array.split(data, split_dim, size_splits)


def concat(raw_tensors, axis):
    """
    concat shapes at axis,  support int8, uint8, int16, int32 float16, float32
    Parameters
    ----------
    raw_tensors : list of tensors
    axis : concat axis
    Returns
    -------
    concat tensor :
    """
    return array.concat(raw_tensors, axis)


def inplace_add(lhs, inplace_ids, rhs):
    """
    calculate inplace add: computes lhs[inplace_ids, :] += rhs; return lhs.

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    inplace_ids : a vector. Indices into the left-most dimension of lhs.

    rhs : wrapped_tensor or tvm.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : computes lhs[inplace_ids, :] += rhs; return lhs.
    """
    return inplace.inplace_add(lhs, inplace_ids, rhs)


def inplace_sub(lhs, inplace_ids, rhs):
    """
    calculate inplace sub: computes lhs[inplace_ids, :] -= rhs; return lhs.

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    inplace_ids : a vector. Indices into the left-most dimension of lhs.

    rhs : wrapped_tensor or tvm.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : computes lhs[inplace_ids, :] -= rhs; return lhs.
    """
    return inplace.inplace_sub(lhs, inplace_ids, rhs)


def inplace_update(lhs, inplace_ids, rhs):
    """
    calculate inplace add: computes lhs[inplace_ids, :] = rhs; return lhs.

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    inplace_ids : a vector. Indices into the left-most dimension of lhs.

    rhs : wrapped_tensor or tvm.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : computes lhs[inplace_ids, :] = rhs; return lhs.
    """
    return inplace.inplace_update(lhs, inplace_ids, rhs)


def pooling2d(tensor_in, window, stride, pooling_mode, padding_mode="SAME",
              pad=(0, 0, 0, 0), dilation=(1, 1), data_mode=1, ceil_mode=0,
              fusion_params=None, impl_mode="high_performance"):
    """
    :params:
    :tensor_in: input tensor
    :window: input window
    :pooling_mode: can be MAX, AVG, GAP, GMP
    :padding_mode: can be SAME, VALID
    :pad: padT, padB, padL, padR
    :dilation: params to be reserved, use default value
    :stride: window move steps in h or w dimension
    :data_mode: can be 0: CAFFE_DATA_MODE, 1: TENSORFLOW_DATA_MODE
    :ceil_mode : caffe round_mode params, 0:CEIL(default), 1:FLOOR
    :return: pooling result
    """
    return pooling2d_compute.pooling2d(tensor_in, window, stride, pooling_mode,
                                       padding_mode, pad, dilation, data_mode,
                                       ceil_mode, fusion_params, impl_mode)


def pooling3d(tensor_in, window, stride, padding_mode="SAME",
              pads=(0, 0, 0, 0, 0, 0),
              pooling_mode="MAX", dilation=(1, 1, 1), ceil_mode=0):
    """
    :params:
    :tensor_in: input tensor
    :window: input window
    :stride: window move steps in d/h/w dimension
    :padding_mode: can be SAME, VALID
    :pads: padFT, padBK,padT,padB,padL,padR, used for caffe,all zero with tf
    :pooling_mode: can be MAX, (AVG, GAP, GMP -- Not support yet)
    :dilation: params to be reserved, use default value
    :ceil_mode : caffe round_mode params, 0:CEIL(default), 1:FLOOR
    :return: pooling result
    """
    return pooling3d_compute.pooling3d(tensor_in, window, stride, padding_mode,
                                       pads, pooling_mode, dilation, ceil_mode)


def max_pooling3d_grad_grad(orig_input, orig_output, grad_grad, assist_tensor,
                            ksize, strides, pads=(0, 0, 0, 0, 0, 0),
                            data_format="NDHWC",
                            padding="SAME"):
    """
    orig_input : dict, shape and dtype of input_data,
                 shape is 6 dims, format is NDC1HWC0
    orig_output : dict, result of max_pool3d(orig_input, ksize, ...)
    grad_grad: dict, input grad of grad
    assist_tensor: dict, helper matrix, it's content is 8,7,6,5,4,3,2,1
                if kernel is 2 x 2 x 2
    ksize : list or tuple, the window of max_pool3d,
            only support max_pool3d in D or H or W
    strides : list or tuple, the stride of max_pool3d window,
              only support max_pool3d in D or H or W
    pads : reserved.
    padding : str, the mode of padding, support SAME or VALID
    ceil_mode: reserved
    """
    return pooling3d_max_grad_grad_compute.max_pooling3d_grad_grad(orig_input,
                                                                   orig_output,
                                                                   grad_grad,
                                                                   assist_tensor,
                                                                   ksize,
                                                                   strides,
                                                                   pads,
                                                                   data_format,
                                                                   padding)


def auto_schedule(outs, option=None):
    """Entry of auto-Schedule.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reduce in the format
          of an array of tensors.
    option:
    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return tbe_auto_schedule.auto_schedule(outs, option)


def build(sch, config_map=None):
    """
    :param sch:
    :param config_map:
    :return:
    """
    return tbe_build(sch, config_map)


def classify(ins: list, mode: str, extra_params: Optional[Dict[str, Any]] = None):
    """
    classify according to mode
    :param ins:
    :param mode: support elewise, broadcast, reduce
    :param extra_params: must include keepdims when mode is reduce
    :return:
    """
    return shape_classifier.classify(ins, mode, extra_params)


def var(name, bound=None, dtype="int32", addition=None):
    """
    add var for external
    :param name:
    :param bound:
    :param dtype: such as int32, float16...
    :param addition:
    :return:
    """
    return operation.var(name, bound, dtype, addition)


def var_attr(name, bound=None, dtype="int32", addition=None):
    """
    var attribute
    :param name:
    :param bound:
    :param dtype: such as int32, float16, int32[4]
    :param addition:
    :return:
    """
    return operation.var_attr(name, bound, dtype, addition)


def add_build_arg(key, value):
    """
    add build arg
    :param key:
    :param value:
    :return:
    """
    return operation.add_build_arg(key, value)


def add_exclude_bound_var(var_):
    """
    add exclude bound var
    :param var_:
    :return:
    """
    return operation.add_exclude_bound_var(var_)


def compute(_operator=None):
    """
    generate a ComputeContext instance
    :param _operator:
    :return:
    """
    return operation.compute(_operator)


def schedule(_compute=None):
    """
    generate a ScheduleContext instance
    :param _compute:
    :return:
    """
    return operation.schedule(_compute)


def conv2d_backprop_filter(input_x, out_backprop, filter_sizes, para_dict):
    """
    the DSL interface of conv2d backprop filter compute

    Parameters:
    ----------
    x : the featuremap data, tvm.placeholder, 5HD shape

    out_backprop : the grads data, tvm.placeholder, 5HD shape

    filter_sizes : 4-D shape, specifies the filter sizes

    para_dict:

        strides : 2-D shape, specifies in height and width dimension

        padding : 4-D shape, specifies in up/down/left/right dimension

        dilations : 4-D shape, specifies in batch/channel/height/width dimension

        groups : The number of filter's group. Default value is 1.

        res_dtype : the output data type

    Returns
    -------
    result tensor of conv2d_backprop_filter compute
    """
    return conv2d_dw_compute.conv2d_backprop_filter_compute(input_x, out_backprop, filter_sizes, para_dict)


def conv2d_backprop_input(filters, out_backprop, filter_sizes, input_sizes, para_dict):
    """
    DSL interface of conv2d backprop input

    Parameters
    ----------
    filters : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_sizes : shape of weight, [N, C, H, W]

    input_sizes : shape of dE/dX, [N, C, H, W]

    para_dict:

        strides : list of strides, [strideh, stridew]

        padding : list of padding, [pad_up, pad_down, pad_left, pad_right]

        dilations : list of dilations, [dilation_n, dilation_c, dilation_h, dilation_w]

        res_dtype : dE/dX data type, "float16" by default

        offset_x : offset of x

        offset_w : offset of w

        fusion_para: the l1 fuison para

        kernel_name : cce kernel name

        group_dict : The params of group convolution.

    Returns
    ----------
    dx_ddr: dE/dX tensor
    """
    return conv2d_dx_compute.conv2d_backprop_input_compute(filters, out_backprop, filter_sizes, input_sizes,
                                                           para_dict)


def conv3d_backprop_filter(x, out_backprop, filter_size, para_dict):
    """
    DSL interface of conv3d bp dx

    Parameters
    ----------
    x : the featuremap data, tvm.placeholder, 6hd shape

    out_backprop : the grads data, tvm.placeholder, 6hd shape

    filter_size : 5-D shape, specifies the filter sizes

    para_dict : dict of parameters
        strides : 3-D shape, specifies in depth, height and width dimension
        pads : 6-D shape, specifies in up/down/left/right dimension
        dilations : 5-D shape, specifies in batch/channel/depth/height/width dimension
        res_dtype : the output data type
        kernel_name : conv3d_backprop_filter_cce by default
        group_dict : group of parameters

    Returns
    -------
    result tensor of conv3d_backprop_filter compute
    """
    return conv3d_dw_compute.conv3d_dw(x, out_backprop, filter_size, para_dict)


def conv3d_backprop_input(filter, out_backprop, filter_size, input_size, para_dict):
    """
    DSL interface of conv3d bp dx

    Parameters
    ----------
    filter : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_size : shape of weight, [N, C, D, H, W]

    input_size : shape of dE/dX, [N, D, H, W, C]

    para_dict : dict of parameters
        strides : list of strides, [stridebatch, strided, strideh, stridew, stridechannel]
        pads : list of padding, [pad_front, pad_tail, pad_up, pad_down, pad_left, pad_right]
        dilations : [1, 1, 1, 1, 1] by default
        res_dtype : dE/dX data type, "float16" by default
        kernel_name : conv3d_backprop_input_cce by default
        group_dict : group of parameters

    Returns
    ----------
    dx_ddr: dE/dX tensor
    """
    return conv3d_dx_compute.conv3d_dx(filter, out_backprop, filter_size, input_size, para_dict)


def conv3d(x, filter, filter_size, para_dict):
    """
    conv

    Parameters
    ----------
    x: feature map

    weight: filter

    filter_size : filter_size

    para_dict: dict of params

    Returns
    -------
    tensor : res
    """
    return conv3d_compute.conv3d(x, filter, filter_size, para_dict)


def depthwise_conv2d_backprop_filter(fmap,
                                     dout,
                                     kernel_h,
                                     kernel_w,
                                     stride,
                                     pad,
                                     dilations,
                                     w_dtype,
                                     kernel_name="depthwise_conv2d_compute"):
    """
    compute of depthwise conv2d backprop filter
    
    the interface will be eliminated soon!

    Parameters
    ----------
    fmap : tvm tensor
        feature map tensor in tvm.

    dout : tvm tensor
        dout tensor in tvm.

    kernel_h: int
        height of filter.

    kernel_w: int
        width of filter.

    stride: tuple or list or int
        stride of convolution.

    pad: list
        padding added to each dimension of the input.

    w_dtype: str
        the dtype of dfilter.

    Returns
    -------
    depthwise_dfilter_res: tvm tensor
        the tensor of output.
    """
    return depthwise_conv2d_compute.depthwise_conv2d_backprop_filter_d_compute(
        fmap, dout, kernel_h, kernel_w, stride, pad, dilations, w_dtype, kernel_name)


def depthwise_conv2d_backprop_input(input_shape,
                                    weight,
                                    dout,
                                    weight_sizes,
                                    strides,
                                    pads,
                                    kernel_name="depthwise_conv2d_compute"):
    """
    Computes the gradients of depthwise convolution with respect to the input.

    the interface will be eliminated soon!

    Parameters
    ----------
    input_shape: a list or tuple representing the shape of input,
                6D format [N, C1, 1, H, W, C0]

    weight: a tensor, 5D with shape [C1, Hf*Wf, 1, C0, C0]

    dout: a tensor, 6D format [N, Co1, 1, Ho, Wo, C0]

    weight_sizes: a list or tuple of two ints,
                  the height and width of the weight of the convolution

    strides: a list or tuple of two ints, the stride of the sliding window for
             height and width of the input of the convolution

    pads: padding added to each dimension of the input

    Returns
    -------
    dx_res: compute of the gradients of depthwise convolution
            with respect to the input
    """
    return depthwise_conv2d_compute.depthwise_conv2d_backprop_input_d_compute(
        input_shape, weight, dout, weight_sizes, strides, pads, kernel_name)


def depthwise_conv2d(fmap,
                     weight,
                     depthwise_res_dtype,
                     stride,
                     pad,
                     dilation,
                     para_dict,
                     l1_fusion_para,
                     kernel_name="depthwise_conv2d_compute"):
    """
    algorithm: depthwise_conv2d_compute

    calculating  depthwise convolution compute

    the interface will be eliminated soon!

    Parameters
    ----------
    fmap : feature map placehold
        5-D shape of input tensor [N, C1, H, W, C0]

    weight : filter placehold
        5-D shape of filter tensor [C1, H, W, Co, C0]

    depthwise_res_dtype : dtype of depthwise UB result

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    pad : padding added to each dimension of the input

    dilation : the dilation factor for each dimension of input

    para_dict : bias tensor dict

    Returns
    -------
    depthwise_res : result tensor
       forward depthwise result of out
    """
    return depthwise_conv2d_compute.depthwise_conv2d_compute(fmap, weight, depthwise_res_dtype, stride, pad,
                                                             dilation, para_dict, l1_fusion_para,
                                                             kernel_name)


def dilation(tensor_x, dilations, pads=None, padding_value=0.0):
    """
    dilation_compute
    :param tensor_x: tensor
    :param dilations: list or tuple
    :param pads: list or tuple or None
    :param padding_value: float
    """
    return dilation_compute.dilation_compute(tensor_x, dilations, pads, padding_value)


def gemm(tensor_a, tensor_b, para_dict):
    """
    algorithm: gemm and matmul
    for gemm:
        calculating matrix multiplication, C = alpha_num*A*B+  beta_num*C
    for matmul:
        caculating matrix multiplication with bias, C = A*B + bias

    Parameters:
    tensor_a: the first tensor a

    tensor_b: second tensor b with the same type and shape with a

              If tensor_a/tensor_b is int8/uint8,then L0A must be 16*32,L0B
              must be 32*16.
              If A is transpose , then AShape classification matrix must be
              32*16 in gm/L1,then it is 16*32 in L0A.
              If B is transpose , then BShape classification matrix must be
              16*32 in gm/L1,then it is 32*16 in L0B.

    para_dict:

    Returns result
    """
    return gemm_compute.gemm(tensor_a, tensor_b, para_dict)
