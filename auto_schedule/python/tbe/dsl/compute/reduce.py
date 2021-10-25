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
reduction compute
"""
import warnings

# pylint: disable=import-error
from decorator import decorator
from tbe import tvm
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.testing.dsl_source_info import source_info_decorator
from tbe.common.utils.errormgr import get_error_message

from .cast import _cast
from .math import vmuls
from .util import check_input_tensor_shape
from .util import dsl_support_dtype
from .util import in_dynamic_and_static_unify
from .util import is_cast_support
from .util import reduce_axis_check
from .util import refine_axis
from .util import shape_to_list
from ..base.expr_compare import expr_equal

_VALUE_MAP_IN_REDUCE_ZERO = {
    "reduce_min": {
        "float16": 6.55e04,
        "float32": 3.4e38,
        "int32": 2147483647,
        "int16": 32767,
    },

    "reduce_max": {
        "float16": -6.55e04,
        "float32": -3.4e38,
        "int32": -2147483648,
        "int16": -32768,
    },

    "reduce_sum": {
        "float16": 0,
        "float32": 0,
        "int32": 0,
        "int16": 0,
    },

    "reduce_prod": {
        "float16": 1,
        "float32": 1,
        "int32": 1,
        "int16": 1,
    },
}

NAME_INDEX = [0]


def is_true(expr, dict_args):
    if expr:
        raise RuntimeError(dict_args, get_error_message(dict_args))


# pylint: disable=too-many-branches
@decorator
def _para_check_of_reduce(func, *args, **kwargs):
    '''
    auto cast dectorator.
    Before calling elewise api, check the input tensor is supported by the intr.
    If not supported, casting the input tensor to supported dtype.
    Only static shape support auto cast.
    (On condition that the cast type is supported.
    If the cast type is not supported,raising a RuntimeError).
    '''
    intr = func.__name__

    if intr == "sum":
        intr = "reduce_sum"

    def _is_last_axis(shape, axis):
        local_axis = []
        for i in axis:
            new_axis = i
            if i < 0:
                new_axis = i + len(shape)
            local_axis.append(new_axis)

        return len(shape) - 1 in local_axis

    def _check_dynamic_dtype(raw_tensor, intr, supported_dtypes, is_last_axis):
        """
        check dtype for dynamic shape
        """
        if not is_last_axis:
            supported_dtypes.append("int32")

        dtype = raw_tensor.dtype

        is_true(dtype not in supported_dtypes,
                {"errCode": "E90002",
                "detailed_cause": "[%s] do not support [%s] in [%s] !" % (intr,
                                                                          dtype,
                                                                          get_soc_spec("SOC_VERSION"))})

    if len(args) == 3 or len(args) == 4:
        is_true(not isinstance(args[0], tvm.tensor.Tensor),
                {"errCode": "E90002",
                "detailed_cause": "The first input type must be [%s]" \
                                          ", while type is [%s]" \
                                          % ('tvm.tensor', type(args[0]))})

        raw_tensor = args[0]
        axis = args[1]
        keepdims = args[2]
        priority_flag = False
        if len(args) == 4:
            priority_flag = args[3]

        if isinstance(axis, (tuple, list)):
            axis = axis
        else:
            axis = [axis]

        shape_len = len(raw_tensor.shape)
        axis = reduce_axis_check(shape_len, axis)

        is_last_axis = _is_last_axis(raw_tensor.shape, axis)

        supported_dtypes = dsl_support_dtype(intr)
        is_true(not supported_dtypes,
                {"errCode": "E90002",
                "detailed_cause": "[%s] is not supported!" % intr})
        # dynamic shape do not perform auto cast
        if in_dynamic_and_static_unify():
            _check_dynamic_dtype(raw_tensor, intr, supported_dtypes, is_last_axis)
            return func(raw_tensor, axis, keepdims)

        return func(raw_tensor, axis, keepdims)

    return func(*args, **kwargs)


# pylint: disable=redefined-builtin
@source_info_decorator()
@_para_check_of_reduce
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
    return _single_reduce_op(raw_tensor, axis, "reduce_sum", keepdims)


@source_info_decorator()
@_para_check_of_reduce
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
    return _single_reduce_op(raw_tensor, axis, "reduce_min", keepdims)


# pylint: disable=unused-argument
@source_info_decorator()
@_para_check_of_reduce
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
    return _single_reduce_op(raw_tensor, axis, "reduce_max", keepdims)


@source_info_decorator()
@_para_check_of_reduce
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
    return _single_reduce_op(raw_tensor, axis, "reduce_prod", keepdims)


def _single_reduce_op(input_tensor,  # pylint: disable=too-many-statements
                      axis, in_op, keepdims=False):
    """
    factory method of single reduce operations
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    """
    is_true(axis is None, {"errCode": "E90001", "detailed_cause": "The axis is None!"})

    def raise_error(message):
        is_true(True, {"errCode": "E90001", "detailed_cause": message})

    check_input_tensor_shape(input_tensor)

    def __reduce_compute(data_shape, axis, tensor, func):
        def compute_func(*indice):
            if contains_zero_axis() and not contains_non_reduce_zero_axis():
                operator = in_op.lower()
                dtype_map = _VALUE_MAP_IN_REDUCE_ZERO.get(operator)
                if dtype_map is None:
                    raise_error("Reduce zero axis does not support operator: {0}.".format(operator))

                dtype = input_tensor.dtype
                value = dtype_map.get(dtype)
                if value is None:
                    raise_error("Reduce zero axis does not support dtype: {0}.".format(dtype))

                return value

            count_indice = 0
            count_reduce = 0
            res_indice = []
            for index in range(len(data_shape)):
                if index not in axis:
                    res_indice.append(indice[count_indice])
                    count_indice += 1
                else:
                    res_indice.append(reduce_axises[count_reduce])
                    count_reduce += 1
                    if keepdims:
                        count_indice += 1

            return func(tensor(*res_indice), axis=reduce_axises)

        reduce_axises = []
        for index, axis_num in enumerate(axis):
            reduce_axises.append(
                tvm.reduce_axis((0, data_shape[axis_num]),
                                name='k' + str(index + 1)))
        res_reshape = []
        for index, shape_l in enumerate(data_shape):
            if index not in axis:
                res_reshape.append(shape_l)
            else:
                if keepdims:
                    res_reshape.append(1)

        # all axis reduce, the dim is 1
        if not res_reshape:
            res_reshape.append(1)

        name = "reduce_" + str(NAME_INDEX[0])
        NAME_INDEX[0] += 1

        reduce_res = tvm.compute(res_reshape, compute_func, name=name)
        return reduce_res

    def contains_zero_axis():
        for _i, dim in enumerate(input_tensor.shape):
            if expr_equal(dim, 0):
                return True
        return False

    def contains_non_reduce_zero_axis():
        for _i, dim in enumerate(input_tensor.shape):
            if expr_equal(dim, 0) and _i not in axis:
                return True
        return False

    def __get_reduce_fun(in_op):
        if in_op.lower() == "reduce_min":
            reduce_func = tvm.min
        elif in_op.lower() == "reduce_max":
            reduce_func = tvm.max
        elif in_op.lower() == "reduce_sum":
            reduce_func = tvm.sum
        elif in_op.lower() == "reduce_prod":
            reduce_func = tvm.prod
        else:
            dict_args = {}
            dict_args["errCode"] = "E90003"
            dict_args["detailed_cause"] = "Not Support yet for op [%s]. " \
                                          "in_op must be reduce_min, " \
                                          "reduce_max, reduce_sum or reduce_prod" % in_op
            raise RuntimeError(dict_args, get_error_message(dict_args))
        return reduce_func

    reduce_func = __get_reduce_fun(in_op)

    op_tensor = input_tensor
    shape = shape_to_list(op_tensor.shape)
    res_axis = refine_axis(axis, shape)

    if keepdims:
        axis = res_axis[:]
        if not in_dynamic_and_static_unify():
            axis_for_loop = axis.copy()
            for index in axis_for_loop:
                if int(input_tensor.shape[index]) == 1:
                    axis.remove(index)
    if not axis:
        res = vmuls(input_tensor, tvm.const(1, dtype=input_tensor.dtype))
        return res

    for i in axis:
        is_last_axis = (i == (len(shape) - 1))
        if is_last_axis:
            break

    with tvm.tag_scope(in_op.lower()):
        res = __reduce_compute(shape, axis, op_tensor, reduce_func)

    return res


@decorator
def _auto_cast_of_tuple_reduce(func, *args, **kwargs):
    '''
    auto cast dectorator.
    Before calling elewise api, check the input tensor is supported by the intr.
    If not supported, casting the input tensor to supported dtype.
    (On condition that the cast type is supported.
    If the cast type is not supported,raising a RuntimeError).
    '''
    func_name = func.__name__
    supported_types = ("float16", "float32")
    is_true(func_name != "tuple_sum", {"errCode": "E90001",
            "detailed_cause": "function name [%s] must be tuple_sum" % func_name})

    def _is_last_axis(shape, axis):
        if isinstance(axis, (tuple, list)):
            local_axis = axis
        else:
            local_axis = [axis]
        return len(shape) - 1 in local_axis

    def _check_tensor(tensor_list):
        is_true(len(tensor_list) != 2, {"errCode": "E90001",
                "detailed_cause": "Tuple reduce input tensors must be 2. " \
                                          "while is [%s]" % len(tensor_list)
                                          })
        shape1 = shape_to_list(tensor_list[0].shape)
        shape2 = shape_to_list(tensor_list[1].shape)
        is_true(shape1 != shape2, {"errCode": "E90001",
                "detailed_cause": "Tuple reduce input tensors must " \
                                          "have same shape. while shape1 is [%s], " \
                                          "shape2 is [%s]" % (shape1, shape2)})

    def _deal_tensor_dtype(raw_tensor, supported_types):
        dtype = raw_tensor.dtype
        if func_name == "tuple_sum" and not _is_last_axis(raw_tensor.shape,
                                                          axis):
            supported_types = supported_types + ("int32",)
        dealed_tensor = raw_tensor
        if dtype not in supported_types:
            if "float32" in supported_types and is_cast_support(dtype,
                                                                "float32"):
                dealed_tensor = _cast(raw_tensor, "float32")
            else:
                dealed_tensor = _cast(raw_tensor, "float16")
        return dealed_tensor

    if len(args) == 3:
        is_true(not isinstance(args[0], (tuple, list)), {"errCode": "E90001",
                "detailed_cause": "The first input type must be list" \
                                          " or tuple, while type is [%s]" % type(args[0])})
        raw_tensor_list = args[0]
        axis = args[1]
        keepdims = args[2]

        _check_tensor(raw_tensor_list)

        temp_tensor_list = []
        for raw_tensor in raw_tensor_list:
            temp_tensor = _deal_tensor_dtype(raw_tensor, supported_types)
            temp_tensor_list.append(temp_tensor)

        return func(temp_tensor_list, axis, keepdims)

    return func(*args, **kwargs)


@source_info_decorator()
@_auto_cast_of_tuple_reduce
def tuple_sum(input_tensor_list, axis, keepdims=False):
    """
    calculate sum of raw_tensor, only support float16
    Parameters
    ----------
    input_tensor_list : wrapped_tensor or tvm.tensor list that each tensor has same reduce operation
    axis : int or list
        reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    Returns
    -------
    res : wrapped_tensor
    """
    warnings.warn("tuple_sum is expired, please replace it with the func sum",
                  DeprecationWarning)
    return _tuple_reduce_op(input_tensor_list, axis, "tuple_reduce_sum",
                            keepdims)


def _tuple_reduce_op(input_tensor_list, axis, in_op, keepdims=False):
    """
    factory method of tuple reduce operations
    keepdims : if true, retains reduced dimensions with length 1, default value is None
    """
    is_true(axis is None, {"errCode": "E90001", "detailed_cause": "The axis is None!"})

    check_input_tensor_shape(input_tensor_list[0])

    if axis in ((), []):
        res = []
        for tensor in input_tensor_list:
            temp_res = vmuls(tensor, tvm.const(1, dtype=tensor.dtype))
            res.append(temp_res)
        return res

    def __tuple_reduce_compute(data_shape, axis, tensor_list, func):
        def compute_func(*indice):
            """
            compute_func
            """
            count_indice = 0
            count_reduce = 0
            res_indice = []
            for index in range(len(data_shape)):
                if index not in axis:
                    res_indice.append(indice[count_indice])
                    count_indice += 1
                else:
                    res_indice.append(reduce_axises[count_reduce])
                    count_reduce += 1
                    if keepdims:
                        count_indice += 1

            return func(
                (tensor_list[0](*res_indice), tensor_list[1](*res_indice)),
                axis=reduce_axises)

        reduce_axises = []
        for index, axis_num in enumerate(axis):
            reduce_axises.append(
                tvm.reduce_axis((0, data_shape[axis_num]),
                                name='k' + str(index + 1)))
        res_reshape = []
        for index, shape_l in enumerate(data_shape):
            if index not in axis:
                res_reshape.append(shape_l)
            else:
                if keepdims:
                    res_reshape.append(1)

        # all axis reduce, the dim is 1
        if not res_reshape:
            res_reshape.append(1)

        name = "reduce_" + str(NAME_INDEX[0])
        NAME_INDEX[0] += 1

        reduce_res = tvm.compute(res_reshape, compute_func, name=name)
        return reduce_res

    tuple_sum_func = tvm.comm_reducer(lambda x, y: (x[0] + y[0], x[1] + y[1]),
                                      lambda t0, t1: (tvm.const(0, dtype=t0),
                                                      tvm.const(0, dtype=t1)),
                                      name="tuple_sum")

    if in_op.lower() == "tuple_reduce_sum":
        reduce_func = tuple_sum_func
    else:
        dict_args = {}
        dict_args["errCode"] = "E90003"
        dict_args["detailed_cause"] = "Not Support yet for op [%s], " \
                                      "in_op must be tuple_reduce_sum" % in_op
        raise RuntimeError(dict_args, get_error_message(dict_args))

    op_tensor = input_tensor_list[0]
    shape = shape_to_list(op_tensor.shape)
    res_axis = refine_axis(axis, shape)
    for i in res_axis:
        is_last_axis = (i == (len(shape) - 1))
        if is_last_axis:
            break

    with tvm.tag_scope(in_op.lower()):
        res = __tuple_reduce_compute(shape, res_axis, input_tensor_list,
                                     reduce_func)

    return res
