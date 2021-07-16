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
dilation compute
"""
import math
import functools
import operator

from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.utils.errormgr import error_manager_util


def shape_align(shape, dtype):
    """
    align the last dim to 32Bit
    :param shape: tuple or list
    :param dtype: data type, support int8, float16, float32
    :return: list; aligned shape
    """
    align_factor = tbe_platform.BLOCK_REDUCE_INT8 if dtype == "int8" else tbe_platform.BLOCK_REDUCE
    res = [i for i in shape]
    res[-1] = math.ceil(res[-1] / align_factor) * align_factor
    return res


def calc_minimum_ub(shape, dilations, dtype):
    """
    calculate the minimum ub size needed according to dilations
    :param shape: tuple or list
    :param dilations: tuple or list
    :param dtype: str
    :return: int
    """
    shape_aligned = shape_align(shape, dtype)
    highest_dilate = get_first_axis_need_dilate(dilations)
    return calc_space_of_ub(shape_aligned, dilations, highest_dilate, dtype)


def get_first_axis_need_dilate(dilations):
    """
    get the highest dim need to be dilated, that is the index of the
    first element which is more than 1.
     for example: for input dilations = (1, 1, 3, 4, 1), result is 2
    :param dilations: list or tuple
    :return: int
    """
    axis = len(dilations) - 1
    for i, value in enumerate(dilations):
        if int(value) > 1:
            axis = i
            break
    return axis


def calc_space_of_ub(shape, dilations, index, dtype):
    """
    calculate the ub_size of shape and its dilation from index
    :param shape: list or tuple
    :param dilations: list or tuple
    :param index: int, index of shape
    :param dtype: string, data type
    :return: int
    """
    shape_dilated = _calc_dilated_shape(shape, dilations)
    num_element = functools.reduce(operator.mul, shape[index:]) + \
                  functools.reduce(operator.mul, shape_dilated[index:])
    type_size_map = {
        "int8": 1,
        "float16": 2,
        "float32": 4
    }
    return type_size_map[dtype] * num_element


def _calc_dilated_shape(shape, dilations):
    """
    calculate dilated shape
    :param shape:list or tuple
    :param dilations: list or tuple
    :return: list, dilated shape
    """
    return list(map(lambda a, b: (a - 1) * b + 1, shape, dilations))


def _param_check(tensor_x, dilations):
    """
    check params
    :param tensor_x: tvm.tensor
    :param dilations: list or tuple
    """
    if tensor_x is None:
        args_dict = {
            "errCode": "E60001",
            "param_name": "x"
        }
        raise RuntimeError(
            args_dict,
            error_manager_util.get_error_message(args_dict)
        )
    if tensor_x.shape is None or len(tensor_x.shape) == 0:
        args_dict = {
            "errCode": "E60029",
            "param_name": "x",
            "key": "shape"
        }
        raise RuntimeError(
            args_dict,
            error_manager_util.get_error_message(args_dict)
        )
    if tensor_x.dtype is None:
        args_dict = {
            "errCode": "E60029",
            "param_name": "x",
            "key": "dtype"
        }
        raise RuntimeError(
            args_dict,
            error_manager_util.get_error_message(args_dict)
        )
    dtype = tensor_x.dtype
    shape_x = [i.value for i in tensor_x.shape]
    shape_x_len = len(shape_x)
    if shape_x_len != len(dilations):
        args_dict = {
            "errCode": "E60002",
            "attr_name": "dim",
            "param1_name": "x",
            "param2_name": 'dilations',
            "param1_value": "{}".format(shape_x_len),
            "param2_value": "{}".format(len(dilations))
        }
        raise RuntimeError(
            args_dict,
            error_manager_util.get_error_message(args_dict)
        )
    if not all(list(value > 0 and type(value) == int for value in dilations)):
        args_dict = {
            "errCode": "E60038",
            "desc": "Elements in dilations should be positive integer"
        }
        raise RuntimeError(
            args_dict,
            error_manager_util.get_error_message(args_dict)
        )
    if dtype not in ("int8", "float16", "float32"):
        args_dict = {
            "errCode": "E60005",
            "param_name": "x",
            "expected_dtype": "(int8, float16, float32)",
            "dtype": "{}".format(dtype)
        }
        raise RuntimeError(
            args_dict,
            error_manager_util.get_error_message(args_dict)
        )
    ub_size_need = calc_minimum_ub(shape_x, dilations, dtype)
    ub_size = tbe_platform_info.get_soc_spec("UB_SIZE")
    if ub_size_need > ub_size:
        args_dict = {
            "errCode": "E60038",
            "desc": "input is too large, the minimum space may exceed UB_Buffer"
        }
        raise RuntimeError(
            args_dict,
            error_manager_util.get_error_message(args_dict)
        )


def dilation_compute(tensor_x, dilations, pads=None, padding_value=0.0):
    """
    dilation_compute
    :param tensor_x: tensor
    :param dilations: list or tuple
    :param pads: list or tuple or None
    :param padding_value: float
    """
    _param_check(tensor_x, dilations)
    shape_x = [i.value for i in tensor_x.shape]
    dtype = tensor_x.dtype
    aligned_shape_x = shape_align(shape_x, dtype)
    shape_dilated = _calc_dilated_shape(aligned_shape_x, dilations)
    padding_default = tvm.compute(
        shape_dilated,
        lambda *indices: tvm.convert(padding_value).astype(dtype),
        name="padding_default"
    )
    ub_x = tvm.compute(
        aligned_shape_x,
        lambda *indices: tvm.select(
            tvm.all(indices[-1] < shape_x[-1]),
            tensor_x(*indices),
            tvm.const(0, dtype=dtype)
        ),
        name="ub_x"
    )
    ub_dilated_x = tvm.compute(
        shape_dilated,
        lambda *indices: tvm.select(
            tvm.all(*list((indices[i] % dilations[i] == 0) for i in range(len(indices)))),
            ub_x(*list(indices[i] // dilations[i] for i in range(len(indices)))),
            padding_default(*indices)
        ),
        name="ub_dilated_x"
    )
    shape_out = shape_dilated[:]
    shape_out[-1] = shape_x[-1]
    dilated_x = tvm.compute(
        shape_out,
        lambda *indices: ub_dilated_x(*indices),
        name="dilated_x",
        tag="dilation"
    )
    return dilated_x
