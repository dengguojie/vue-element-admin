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
cast compute
"""
from te import platform as cceconf
from te import tvm

from . import elewise_compute
from .util import DTYPE_MAP
from .util import check_input_tensor_shape
from .util import is_cast_support
from .util import shape_to_list

NAME_INDEX = [0]


# 'pylint: disable=R0914
def cast_to(data, dtype, fp162int8_flag=True):
    """
    a wrapped cast operations , cast data to the type of dtype

    Parameters
    ----------
    data : tvm.tensor
        tensors need to change dtype

    dtype : string
        dst dtype need to cast to

    fp162int8_flag : bool
        before fp16->int8/uint8, the data is all integer or not. default value
        is False.

    Returns
    -------
    tensor : tvm.tensor
    """
    if isinstance(data, tvm.tensor.Tensor):
        data_dtype = getattr(data, 'dtype')
    else:
        raise RuntimeError("The cast input type must be tvm.tensor")
    check_input_tensor_shape(data)

    if (data_dtype.lower() == "float32") and (dtype == "int32"):
        fp32_max = tvm.const(2147483648, dtype="float32")
        fp32_min = tvm.const(2 ** (-31), dtype="float32")
        data1 = round_to(data, 0.5, -0.5)
        new_data = elewise_compute.vmuls(data1, fp32_max)
        tmp2 = elewise_compute.vabs(new_data)
        tmp3 = elewise_compute.vadds(tmp2, fp32_min)
        fp32_res = elewise_compute.vmul(new_data, elewise_compute.vrec(tmp3))
        if not cceconf.cce_conf.\
                api_check_support("te.lang.dynamic.round", "float32"):
            fp32_res = _cast(fp32_res, dst_dtype="float16")
        sign_res = round(fp32_res)  # get the sign data[-1,0,1]

        abs_data = elewise_compute.vabs(data)
        if not cceconf.cce_conf.\
                api_check_support("te.lang.dynamic.floor", "float32"):
            abs_data = _cast(abs_data, dst_dtype="float16")
        floor_data = floor(abs_data)  # get the floor data
        res = elewise_compute.vmul(floor_data, sign_res)
        return res

    if (data_dtype.lower() == "float16") and (dtype == "int32"):
        fp16_max = tvm.const(32768, dtype="float16")
        fp16_min = tvm.const(2 ** (-15), dtype="float16")

        data1 = round_to(data, 0.5, -0.5)

        new_data = elewise_compute.vmuls(data1, fp16_max)
        tmp2 = elewise_compute.vabs(new_data)
        tmp3 = elewise_compute.vadds(tmp2, fp16_min)
        fp16_res = elewise_compute.vmul(new_data, elewise_compute.vrec(tmp3))
        sign_res = round(fp16_res)

        floor_data = floor(elewise_compute.vabs(data))
        res = elewise_compute.vmul(floor_data, sign_res)
        return res

    if (data_dtype.lower() == "float16") and (dtype in (
            'int8', 'uint8')) and not fp162int8_flag:
        fp16_half = tvm.const(-0.5, dtype="float16")
        data = elewise_compute.vadds(data, fp16_half)

    if data_dtype == dtype:
        return data

    return _cast(data, dst_dtype=dtype)


def round_to(data, max_value, min_value):
    """
    round data to [min_value,max_value]

    Parameters
    ----------
    data : tvm.tensor
        tensors need to change dtype

    max_value : float
        the range of res

    min_value : float
        the range of res

    Returns
    -------
    tensor : tvm.tensor ,elements in tensor is in range [min_value,max_value]
    """
    if isinstance(data, tvm.tensor.Tensor):
        check_input_tensor_shape(data)
    data_tmp = elewise_compute.vmuls(data, 0)
    data_min = elewise_compute.vadds(data_tmp, min_value)
    data_max = elewise_compute.vadds(data_tmp, max_value)
    data1 = elewise_compute.vmax(data, data_min)
    data1 = elewise_compute.vmin(data1, data_max)
    return data1


def _cast(raw_tensor, dst_dtype):
    """
    cast tensor from src_type to dst_dtype, only support float32 to float16,
    float16 to float32, float16 to int8, int8 to float16,
    float16 to uint8, uint8 to float16

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    dst_dtype : destinatin type

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    src_dtype = raw_tensor.dtype
    if dst_dtype.lower() == src_dtype.lower():
        return raw_tensor

    if not is_cast_support(src_dtype.lower(), dst_dtype.lower()):
        if is_cast_support(src_dtype.lower(), "float32") and \
                is_cast_support("float32", dst_dtype.lower()):
            raw_tensor = _cast_op(raw_tensor, "float32", 'elewise_single_cast')
        elif is_cast_support(src_dtype.lower(), "float16") and \
                is_cast_support("float16", dst_dtype.lower()):
            raw_tensor = _cast_op(raw_tensor, "float16", 'elewise_single_cast')
        else:
            raise RuntimeError("Unsupported cast type!")

    return _cast_op(raw_tensor, dst_dtype, 'elewise_single_cast')


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
    dst_dtype = "int32"
    return _cast_op(raw_tensor, dst_dtype, "elewise_single_ceil")


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
    dst_dtype = "int32"
    return _cast_op(raw_tensor, dst_dtype, "elewise_single_floor")


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
    src_dtype = raw_tensor.dtype.lower()
    cast_type = DTYPE_MAP[src_dtype] + "2s32z"
    is_support = cceconf.intrinsic_check_support("Intrinsic_vconv", cast_type)
    if not is_support:
        raise RuntimeError("the target platform is not support %s trunc"
                           % src_dtype)

    dst_dtype = "int32"
    return _cast_op(raw_tensor, dst_dtype, "elewise_single_trunc")


# 'pylint: disable=redefined-builtin
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
    dst_dtype = "int32"
    return _cast_op(raw_tensor, dst_dtype, "elewise_single_round")


def round_d(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with rounding method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    src_dtype = raw_tensor.dtype.lower()
    cast_type = DTYPE_MAP[src_dtype] + "2s32a"
    is_support_round_d = cceconf.intrinsic_check_support("Intrinsic_vconv",
                                                         cast_type)
    if not is_support_round_d:
        raise RuntimeError("the target platform is not support %s round"
                           % src_dtype)

    dst_dtype = "int32"
    return _cast_op(raw_tensor, dst_dtype, "elewise_single_round_d")


def _cast_op(input_tensor, output_dtype, op_type):
    """
    factory method of single elewise operations
    """
    tensor = input_tensor
    shape = shape_to_list(tensor.shape)
    check_input_tensor_shape(shape)

    if op_type == "elewise_single_cast":
        lambda_func = lambda *i: tensor(*i).astype(output_dtype)
    elif op_type == "elewise_single_ceil":
        lambda_func = lambda *i: tensor(*i).astype(output_dtype)
    elif op_type == "elewise_single_floor":
        lambda_func = lambda *i: tensor(*i).astype(output_dtype)
    elif op_type == "elewise_single_trunc":
        lambda_func = lambda *i: tensor(*i).astype(output_dtype)
    elif op_type == "elewise_single_round":
        lambda_func = lambda *indice: tensor(*indice).astype(output_dtype)
    elif op_type == "elewise_single_round_d":
        lambda_func = lambda *i: tensor(*i).astype(output_dtype)
    else:
        raise RuntimeError("operation %s not support yet" % op_type)

    name = op_type.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    op_type = op_type + "|not_auto_cast"

    with tvm.tag_scope(op_type):
        tmp = tvm.compute(shape, lambda_func, name=name)

    return tmp
