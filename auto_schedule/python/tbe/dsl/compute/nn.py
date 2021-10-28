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
nn
"""
from __future__ import absolute_import

from tbe import tvm
from tbe.common.platform import intrinsic_check_support
from tbe.common.testing.dsl_source_info import source_info_decorator
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base.expr_compare import expr_equal as equal

from .math import __multiple_elewise_op
from .math import __single_elewise_op
from .math import _cast_tensors_for_instr
from .math import _check_multi_compute_pattern
from .math import vadd
from .math import vadds
from .math import vmax
from .math import vmin
from .math import vmuls
from .math import vsub
from .util import check_input_tensor_shape
from .util import dtype_check_decorator
from .util import in_dynamic_and_static_unify
from .util import judge_var
from .util import shape_to_list

NAME_INDEX = [0]


@source_info_decorator()
@dtype_check_decorator
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
    if not isinstance(tensor_1, tvm.tensor.Tensor):
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" % (
                                      'tvm.tensor', type(tensor_1))
        raise RuntimeError(dict_args, get_error_message(dict_args))
    if not isinstance(tensor_2, tvm.tensor.Tensor):
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The third input type must be [%s], " \
                                      "while type is [%s]" % (
                                      'tvm.tensor', type(tensor_2))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return __multiple_elewise_op(tensor_0, tensor_1, tensor_2,
                                 "elewise_multiple_maddrelu")


@source_info_decorator()
@dtype_check_decorator
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
    if not isinstance(lhs, tvm.tensor.Tensor):
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The lhs input type must be [%s], " \
                                      "while type is [%s]" \
                                      % ('tvm.tensor', type(lhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if not isinstance(rhs, tvm.tensor.Tensor):
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The rhs input type must be [%s], " \
                                      "while type is [%s]" \
                                      % ('tvm.tensor', type(rhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if lhs.dtype != rhs.dtype:
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "dtype must be the same, " \
                                      "while lhs is %s, rhs is %s" % (
                                      lhs.dtype, rhs.dtype)
        raise RuntimeError(dict_args, get_error_message(dict_args))

    is_current_chip_support = intrinsic_check_support("Intrinsic_vaddrelu")
    if not is_current_chip_support:
        add = vadd(lhs, rhs)
        res = vrelu(add)
        return res

    shape = lhs.shape
    op_name = "elewise_binary_addrelu"
    lambda_func = lambda *indice: tvm.relu(lhs(*indice) + rhs(*indice))

    name = op_name.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    with tvm.tag_scope(op_name):
        tmp = tvm.compute(shape, lambda_func, name=name)

    return tmp


@source_info_decorator()
@dtype_check_decorator
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
    if not isinstance(lhs, tvm.tensor.Tensor):
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The lhs input type must be [%s], " \
                                      "while type is [%s]" \
                                      % ('tvm.tensor', type(lhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if not isinstance(rhs, tvm.tensor.Tensor):
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The rhs input type must be [%s], " \
                                      "while type is [%s]" \
                                      % ('tvm.tensor', type(rhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if lhs.dtype != rhs.dtype:
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "dtype must be the same, " \
                                      "while lhs is %s, rhs is %s" % (
                                          lhs.dtype, rhs.dtype)
        raise RuntimeError(dict_args, get_error_message(dict_args))

    is_current_chip_support = intrinsic_check_support("Intrinsic_vsubrelu")
    if not is_current_chip_support:
        sub = vsub(lhs, rhs)
        res = vrelu(sub)
        return res

    shape = lhs.shape
    op_name = "elewise_binary_subrelu"
    lambda_func = lambda *indice: tvm.relu(lhs(*indice) - rhs(*indice))

    name = op_name.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    with tvm.tag_scope(op_name):
        tmp = tvm.compute(shape, lambda_func, name=name)

    return tmp


@source_info_decorator()
@dtype_check_decorator
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
    _is_current_chip_support_vaddrelu = intrinsic_check_support("Intrinsic_vaddrelu")
    vaddrelu_pattern = _check_multi_compute_pattern(("elewise_binary_add",), raw_tensor)
    if _is_current_chip_support_vaddrelu and vaddrelu_pattern:
        input_tensor = tuple(raw_tensor.op.input_tensors)
        return vaddrelu(*input_tensor)

    _is_current_chip_support_vsubrelu = intrinsic_check_support("Intrinsic_vsubrelu")
    vsubrelu_pattern = _check_multi_compute_pattern(("elewise_binary_sub",), raw_tensor)
    if _is_current_chip_support_vsubrelu and vsubrelu_pattern:
        input_tensor = tuple(raw_tensor.op.input_tensors)
        return vsubrelu(*input_tensor)

    dtype = raw_tensor.dtype

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_relu')


@source_info_decorator()
@dtype_check_decorator
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
    dtype = raw_tensor.dtype
    shape = raw_tensor.shape

    if judge_var(alpha) == "tvm_const":
        if alpha.dtype != dtype:
            dict_args = {}
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The dtype of alpha [%s] " \
                                          "must be equal to raw_tensor's [%s]"\
                                          % (alpha.dtype, dtype)
            raise RuntimeError(dict_args, get_error_message(dict_args))
        alpha_value = alpha.value
    elif judge_var(alpha) == "python_const":
        alpha_value = alpha
        alpha = tvm.const(alpha, dtype=dtype)
    else:
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" % \
                                      ('tvm_const or python_const', type(alpha))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    is_current_chip_support = intrinsic_check_support("Intrinsic_vlrelu")
    if not is_current_chip_support:
        if alpha_value == 0:
            if dtype in ("float32", "int32"):
                tensor_zero = broadcast(tvm.const(0, dtype), shape)
                data_res = vmax(raw_tensor, tensor_zero)
            else:
                data_res = vrelu(raw_tensor)

            return data_res

        muls_tmp = vmuls(raw_tensor, alpha)
        if alpha_value <= 1:
            res = vmax(raw_tensor, muls_tmp)
        else:
            res = vmin(raw_tensor, muls_tmp)

        return res

    op_name = "elewise_single_lrelu"
    lambda_func = lambda *indice: tvm.lrelu(raw_tensor(*indice), alpha)

    name = op_name.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    with tvm.tag_scope(op_name):
        tmp = tvm.compute(shape, lambda_func, name=name)

    return tmp


@source_info_decorator()
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
    if isinstance(data, tvm.tensor.Tensor):
        check_input_tensor_shape(data)
    tensor_vmuls = _cast_tensors_for_instr("vmuls", [data, 0])
    data_tmp = vmuls(tensor_vmuls[0], tensor_vmuls[1])
    tensor_vadds = _cast_tensors_for_instr("vadds", [data_tmp, min_value])
    data_min = vadds(tensor_vadds[0], tensor_vadds[1])
    tensor_vadds_tmp = _cast_tensors_for_instr("vadds", [data_tmp, max_value])
    data_max = vadds(tensor_vadds_tmp[0], tensor_vadds_tmp[1])
    tensor_vmax = _cast_tensors_for_instr("vmax", [data, data_min])
    data1 = vmax(tensor_vmax[0], tensor_vmax[1])
    tensor_vmin = _cast_tensors_for_instr("vmin", [data1, data_max])
    data1 = vmin(tensor_vmin[0], tensor_vmin[1])
    return data1


@source_info_decorator()
@dtype_check_decorator
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
    if not isinstance(shape, (list, tuple, tvm.container.Array)):
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "the input parameter shape must be " \
                                      "list or tuple, while type of input is %s" % (type(shape))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if isinstance(var, tvm.tensor.Tensor):
        return _tensor_broadcast(var, shape)

    var_type = judge_var(var)
    tmp_args = var
    if var_type == "python_const":
        if isinstance(tmp_args, float):
            tmp_args = tvm.const(tmp_args, dtype="float16")
        else:
            tmp_args = tvm.const(tmp_args, dtype="int32")

    if not output_dtype:
        output_dtype = tmp_args.dtype

    tmp_args = tmp_args.astype(output_dtype)

    lambda_func = lambda *indice: tmp_args

    name = "broadcast_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    _op = 'broadcast'
    with tvm.tag_scope(_op):
        out = tvm.compute(shape, lambda_func, name=name)
    return out


def _tensor_broadcast(var, shape) -> tvm.tensor.Tensor:
    """
    broadcast tensor to tensor

    Parameters
    ----------
    var : can be tvm.tensor.Tensor

    shape : tensor shape

    Returns
    -------
    wrapped_tensor : broadcast tensor
    """
    tensor = var
    orig_shape = shape_to_list(tensor.shape)
    check_input_tensor_shape(orig_shape)
    if len(orig_shape) > len(shape):
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "Length of original shape must be " \
                                      "smaller than target shape, but src shape is %s, " \
                                      "and dst shape is %s" % (str(orig_shape), str(shape))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    valid_types = (tvm.expr.ConstExpr, tvm.expr.Var, tvm.expr.Max, tvm.expr.Mul)
    difference = len(shape) - len(orig_shape)
    orig_shape = difference * [1] + orig_shape
    check_equal = 0
    is_unknown_broadcast = False
    for src_shape, dst_shape in zip(orig_shape, shape):
        if equal(src_shape, dst_shape):
            check_equal += 1
            continue
        if equal(src_shape, 1):
            continue
        if isinstance(src_shape, valid_types) or \
                isinstance(dst_shape, valid_types):
            is_unknown_broadcast = True
            continue
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "For tensor broadcasting, shape must " \
                                      "be the same or corresponding shape of" \
                                      " src tensor is 1 while src shape is %s," \
                                      " and dst shape is %s" % (str(orig_shape), str(shape))
        raise RuntimeError(dict_args, get_error_message(dict_args))
    if check_equal == len(shape):
        return tensor

    name = "broadcast_tensor_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    if in_dynamic_and_static_unify():
        if is_unknown_broadcast:
            _op = 'unknown_broadcast'
        else:
            _op = 'unified_broadcast'
    else:
        _op = 'broadcast_for_tensor'

    def lambda_func(*indices):
        if _op == 'unknown_broadcast':
            index = []
            for i in range(len(orig_shape)):
                if orig_shape[i] == 1:
                    index.append(0)
                elif equal(orig_shape[i], shape[i]):
                    index.append(indices[i])
                else:
                    index.append(tvm.select(orig_shape[i] == 1, 0, indices[i]))
            return tensor(*(index[difference:]))
        return tensor(*([0 if orig_shape[i] == 1 else
                         indices[i] for i in range(len(orig_shape))][difference:]))

    with tvm.tag_scope(_op):
        out = tvm.compute(shape, lambda_func, name=name)

    return out
