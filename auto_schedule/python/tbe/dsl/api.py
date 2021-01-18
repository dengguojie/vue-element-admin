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
from .compute import cast
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


def ceil(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with ceiling method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.ceil(raw_tensor)


def floor(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with flooring method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.floor(raw_tensor)


def round(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with rounding method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.round(raw_tensor)


def trunc(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with trunc method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.trunc(raw_tensor)


def round_half_up(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with rounding method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return cast.round_half_up(raw_tensor)


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


def vrec(raw_tensor, priority_flag=1):
    """
    calculate vrec(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    priority_flag: priority flag, only support 1(precision), 0(performance)

    Returns
    -------
    wrapped_tensor : vrec(raw_tensor)
    """
    return math.vrec(raw_tensor, priority_flag)


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


def vlog(raw_tensor, priority_flag=0):
    """
    calculate ln(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    priority_flag : priority flag, only support 1(precision) and 0(performance)

    Returns
    -------
    wrapped_tensor : log(raw_tensor)
    """
    return math.vlog(raw_tensor, priority_flag)


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


def vsqrt(raw_tensor, priority_flag=0):
    """
    calculate vsqrt(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    priority_flag: priority flag, only support 1(precision), 0(performance)

    Returns
    -------
    wrapped_tensor : vsqrt(raw_tensor)
    """
    return math.vsqrt(raw_tensor, priority_flag)


def vrsqrt(raw_tensor, priority_flag=0):
    """
    calculate vrsqrt(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vrsqrt(raw_tensor)
    """
    return math.vrsqrt(raw_tensor, priority_flag)


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


def reduce_min(raw_tensor, axis, keepdims=False, priority_flag=False):
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
    return reduce.reduce_min(raw_tensor, axis, keepdims, priority_flag)


def reduce_max(raw_tensor, axis, keepdims=False, priority_flag=False):
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
    return reduce.reduce_max(raw_tensor, axis, keepdims, priority_flag)


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


def classify(ins: list, mode: str):
    """
    classify according to mode
    :param ins:
    :param mode:
    :return:
    """
    return shape_classifier.classify(ins, mode)
