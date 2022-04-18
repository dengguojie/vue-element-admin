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
cast
"""
# 'pylint: disable=import-error
import warnings

import tbe.dsl
from decorator import decorator
from tbe import tvm
from tbe.common.platform import intrinsic_check_support
from tbe.common.platform.platform_info import api_check_support
from tbe.common.testing.dsl_source_info import source_info_decorator
from tbe.common.testing.testing import is_debug_mode
from tbe.common.utils.errormgr import get_error_message

from .util import DTYPE_MAP
from .util import check_input_tensor_shape
from .util import dsl_support_dtype
from .util import get_cast_type
from .util import in_dynamic_and_static_unify
from .util import is_cast_support
from .util import shape_to_list

NAME_INDEX = [0]


@decorator
def _para_check_of_cast(func, *args, **kwargs):
    '''
    cast dectorator used for round ceil floor trunc and round_half_up.
    Before calling elewise api, check the input tensor is supported by the intr.
    If not supported, would rasie up and error
    CAST_MAP:
        "round": "r",
        "floor": "f",
        "trunc": "z",
        "round_half_up": "a",
        "ceil": "c"
    '''
    intr = func.__name__
    if len(args) == 2:
        if not isinstance(args[0], tvm.tensor.Tensor):
            dict_args = {}
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The first input type must be [%s]," \
                                          " while type is [%s]" % ('tvm.tensor', type(args[0]))
            raise RuntimeError(dict_args, get_error_message(dict_args))

        src_dtype = args[0].dtype.lower()
        dst_dtype = args[1]
        cast_type = DTYPE_MAP[src_dtype] + "2" + DTYPE_MAP[dst_dtype]

        supported_dtypes = dsl_support_dtype(intr)
        if not supported_dtypes:
            dict_args = {}
            dict_args["errCode"] = "E90002"
            dict_args["detailed_cause"] = "[%s] is not supported!" % intr
            raise RuntimeError(dict_args, get_error_message(dict_args))

        if cast_type not in supported_dtypes:
            dict_args = {}
            dict_args["errCode"] = "E90002"
            dict_args["detailed_cause"] = "the target platform is not " \
                                          "support %s %s" % (cast_type, intr)
            raise RuntimeError(dict_args, get_error_message(dict_args))

    return func(*args, **kwargs)


def _cast(raw_tensor, dst_dtype, is_auto_cast=True):
    """
    cast tensor from src_type to dst_dtype, only support float32 to float16,
    float16 to float32, float16 to int8, int8 to float16,
    float16 to uint8, uint8 to float16 when shape is dynamic

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
        if is_cast_support(src_dtype.lower(), "float32") and is_cast_support(
                "float32",
                dst_dtype.lower()):
            raw_tensor = _cast_op(raw_tensor, "float32", 'elewise_single_cast')
            src_dtype = raw_tensor.dtype
        elif is_cast_support(src_dtype.lower(), "float16") and is_cast_support(
                "float16",
                dst_dtype.lower()):
            raw_tensor = _cast_op(raw_tensor, "float16", 'elewise_single_cast')
            src_dtype = raw_tensor.dtype
        elif not intrinsic_check_support("Intrinsic_vconv", "deq") and \
                intrinsic_check_support("Intrinsic_vcbd", "s322s16"):
            raw_tensor = _cast_op(raw_tensor, "int16", 'elewise_single_cast')
            src_dtype = raw_tensor.dtype
        else:
            dict_args = {}
            dict_args["errCode"] = "E90002"
            dict_args["detailed_cause"] = "Unsupported cast type! src_dtype is [%s]," \
                                          " dst_dtype is [%s]" % (src_dtype.lower(), dst_dtype.lower())
            raise RuntimeError(dict_args, get_error_message(dict_args))

    # Default cast_type is "cast", while float casting to int32 or int16 is "trunc"
    # float to int8 or uint8 doesn't need to use "trunc" mode
    # Reason: TF uses "trunc" strategy in its cast operator, so maintain consistency with it
    cast_type = "cast"
    if "int" in dst_dtype and "float" in src_dtype and \
            intrinsic_check_support("Intrinsic_vconv",
                                    get_cast_type(src_dtype, dst_dtype) + "z"):
        cast_type = "trunc"
    return _cast_op(raw_tensor, dst_dtype, 'elewise_single_' + cast_type,
                    is_auto_cast=is_auto_cast)


@source_info_decorator()
@_para_check_of_cast
def ceil(raw_tensor, dst_dtype="int32"):
    """
    cast tensor from src_type to dst_dtype with ceiling method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return _cast_op(raw_tensor, dst_dtype, "elewise_single_ceil")


@source_info_decorator()
@_para_check_of_cast
def floor(raw_tensor, dst_dtype="int32"):
    """
    cast tensor from src_type to dst_dtype with flooring method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return _cast_op(raw_tensor, dst_dtype, "elewise_single_floor")


def _floor(raw_tensor):
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


# 'pylint: disable=redefined-builtin
@source_info_decorator()
@_para_check_of_cast
def round(raw_tensor, dst_dtype="int32"):
    """
    cast tensor from src_type to dst_dtype with rounding method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return _cast_op(raw_tensor, dst_dtype, "elewise_single_round")


def _round(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with rounding method for inner use

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    dst_dtype = "int32"
    return _cast_op(raw_tensor, dst_dtype, "elewise_single_round")


@source_info_decorator()
@_para_check_of_cast
def trunc(raw_tensor, dst_dtype="int32"):
    """
    cast tensor from src_type to dst_dtype with trunc method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return _cast_op(raw_tensor, dst_dtype, "elewise_single_trunc")


@source_info_decorator()
@_para_check_of_cast
def round_half_up(raw_tensor, dst_dtype="int32"):
    """
    cast tensor from src_type to dst_dtype with rounding method

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    return _cast_op(raw_tensor, dst_dtype, "elewise_single_round_d")


@source_info_decorator()
def round_d(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with rounding method
    """
    warnings.warn("round_d is expired, please replace it with the func round_half_up",
                  DeprecationWarning)
    return round_half_up(raw_tensor)


def _cast_op(input_tensor, output_dtype, op_type, is_auto_cast=True):
    """
    factory method of single elewise operations
    Owning to the previous compute of cast Only for DaVinci,  there are use
    tvm.* for proving correctness for dsl operator
    """
    tensor = input_tensor
    shape = tensor.shape
    check_input_tensor_shape(shape)
    if is_debug_mode():
        return _cast_op_for_protogenes(tensor, op_type, shape)
    return _cast_op_for_davinci(tensor, is_auto_cast, op_type, shape, output_dtype)


def _cast_op_for_davinci(tensor, is_auto_cast, op_type, shape, output_dtype):
    """
    cast operate for DaVinci
    """
    if op_type == "elewise_single_cast":
        lambda_func = lambda *indice: tensor(*indice).astype(output_dtype)
    elif op_type == "elewise_single_round":
        lambda_func = lambda *indice: tensor(*indice).astype(output_dtype)
    elif op_type == "elewise_single_ceil":
        lambda_func = lambda *indice: tensor(*indice).astype(output_dtype)
    elif op_type == "elewise_single_floor":
        lambda_func = lambda *indice: tensor(*indice).astype(output_dtype)
    elif op_type == "elewise_single_trunc":
        lambda_func = lambda *indice: tensor(*indice).astype(output_dtype)
    elif op_type == "elewise_single_round_d":
        lambda_func = lambda *indice: tensor(*indice).astype(output_dtype)
    else:
        dict_args = {}
        dict_args["errCode"] = "E90003"
        dict_args["detailed_cause"] = "operation %s not support yet." % op_type
        raise RuntimeError(dict_args, get_error_message(dict_args))
    name = op_type.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1
    if not is_auto_cast:
        op_type = op_type + "|not_auto_cast"
    with tvm.tag_scope(op_type):
        tmp = tvm.compute(shape, lambda_func, name=name)
    return tmp


def _cast_op_for_protogenes(tensor, op_type, shape):
    """
    cast operate for protogenes
    In this situation, use tvm.* for cast
    """
    if op_type == "elewise_single_round":
        lambda_func = lambda *indice: tvm.round(tensor(*indice))
    elif op_type == "elewise_single_ceil":
        lambda_func = lambda *indice: tvm.ceil(tensor(*indice))
    elif op_type == "elewise_single_floor":
        lambda_func = lambda *indice: tvm.floor(tensor(*indice))
    elif op_type == "elewise_single_trunc":
        lambda_func = lambda *indice: tvm.trunc(tensor(*indice))
    else:
        dict_args = {}
        dict_args["errCode"] = "E90003"
        dict_args["detailed_cause"] = "operation %s not support yet." % op_type
        raise RuntimeError(dict_args, get_error_message(dict_args))
    name = op_type.split("_")[-1] + "_" + str(NAME_INDEX[0])
    return tvm.compute(shape, lambda_func, name=name)


# 'pylint: disable=too-many-locals, invalid-name
@source_info_decorator()
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
    if isinstance(data, tvm.tensor.Tensor):
        data_dtype = getattr(data, 'dtype')
    else:
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The cast input type must be " \
                                      "tvm.tensor, while is [%s]" % type(data)
        raise RuntimeError(dict_args, get_error_message(dict_args))
    check_input_tensor_shape(data)

    if (data_dtype.lower() == "float32") and (dtype == "int32") and \
            not intrinsic_check_support("Intrinsic_vconv", "f322s32z"):
        fp32_max = tvm.const(2147483648, dtype="float32")
        fp32_min = tvm.const(2 ** (-31), dtype="float32")
        data1 = tbe.dsl.clip(data, 0.5, -0.5)
        new_data = tbe.dsl.vmuls(data1, fp32_max)
        tmp2 = tbe.dsl.vabs(new_data)
        tmp3 = tbe.dsl.vadds(tmp2, fp32_min)
        fp32_res = tbe.dsl.vmul(new_data, tbe.dsl.vrec(tmp3, "high_precision"))
        if not api_check_support("tbe.dsl.round", "float32"):
            fp32_res = _cast(fp32_res, dst_dtype="float16")
        sign_res = _round(fp32_res)  # get the sign data[-1,0,1]
        abs_data = tbe.dsl.vabs(data)
        if not api_check_support("tbe.dsl.floor", "float32"):
            abs_data = _cast(abs_data, dst_dtype="float16")
        floor_data = _floor(abs_data)  # get the floor data
        res = tbe.dsl.vmul(floor_data, sign_res)
        return res

    if (data_dtype.lower() == "float16") and (dtype == "int32") and \
            not intrinsic_check_support("Intrinsic_vconv", "f162s32z"):
        fp16_max = tvm.const(32768, dtype="float16")
        fp16_min = tvm.const(2 ** (-15), dtype="float16")

        data1 = tbe.dsl.clip(data, 0.5, -0.5)

        new_data = tbe.dsl.vmuls(data1, fp16_max)
        tmp2 = tbe.dsl.vabs(new_data)
        tmp3 = tbe.dsl.vadds(tmp2, fp16_min)
        fp16_res = tbe.dsl.vmul(new_data, tbe.dsl.vrec(tmp3, "high_precision"))
        sign_res = _round(fp16_res)

        floor_data = _floor(tbe.dsl.vabs(data))
        res = tbe.dsl.vmul(floor_data, sign_res)
        return res

    if (data_dtype.lower() == "float16") and (dtype in (
            'int8', 'uint8')) and not f1628IntegerFlag:
        fp16_half = tvm.const(-0.5, dtype="float16")
        data = tbe.dsl.vadds(data, fp16_half)

    if data_dtype == dtype:
        return data

    return _cast(data, dst_dtype=dtype, is_auto_cast=False)


@source_info_decorator()
def cast_to_round(data, dtype):
    """
    Parameters
    ----------
    data : tvm.tensor
        tensors need to change dtype

    dtype : string
        dst dtype need to cast to

    Returns
    -------
    tensor : tvm.tensor
    """
    warnings.warn("cast_to_round is expired, please replace it with the func round_half_up and cast_to",
                  DeprecationWarning)
    dtype = dtype.lower()
    if dtype != "int32":
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The cast output dtype must be int32," \
                                      " while is [%s]" % dtype
        raise RuntimeError(dict_args, get_error_message(dict_args))

    src_dtype = data.dtype.lower()
    cast_type = DTYPE_MAP[src_dtype] + "2s32a"
    is_support_round_d = intrinsic_check_support("Intrinsic_vconv", cast_type)
    if not is_support_round_d:
        return cast_to(data, dtype)

    return round_half_up(data)

