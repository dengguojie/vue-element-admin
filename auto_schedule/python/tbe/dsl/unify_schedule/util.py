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
schedule util
"""
from functools import reduce
from operator import mul
from typing import List
from typing import Union

from tbe import tvm
from tbe.common.platform import ASCEND_310
from tbe.common.platform import ASCEND_610
from tbe.common.platform import ASCEND_615
from tbe.common.platform import ASCEND_710
from tbe.common.platform import ASCEND_910
from tbe.common.platform import HI3796CV300CS
from tbe.common.platform import HI3796CV300ES
from tbe.common.platform import SD3403
from tbe.common.platform import ASCEND_920A
from tbe.common.platform import SOC_VERSION
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation
from tbe.dsl.base.expr_compare import expr_equal
from tbe.tvm.expr import Reduce
from tbe.tvm.expr import Var
from tbe.tvm.tensor import PlaceholderOp
from tbe.tvm.tensor import Tensor

from .constants import BROADCAST_INSNS
from .constants import SUPPORT_SCALAR_INSNS
from .constants import NEED_TEMP_SPACE_INSNS
from .constants import VCMP_INSNS
from .constants import VSEL_INSNS
from .constants import VCMPSEL_INSNS
from .constants import NEED_SPACE_WITH_DIFF_TYPE
from .constants import NEED_EXTENT_NODE_INSNS

VAR_BOUND_LIMIT = 2147483647


def is_true(expr, dict_args):
    """
    :param expr: condition
    :param dict_args: error message
    :return: RuntimeError
    """
    if expr:
        raise RuntimeError(dict_args, get_error_message(dict_args))


def shape_to_list(shape):
    """
    :param shape:
    :return:
    """
    shape0 = []
    for i in shape:
        if isinstance(i, tvm.expr.ConstExpr):
            shape0.append(i.value)
        else:
            shape0.append(i)
    return shape0


def get_dsl_insn(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    tag = tensor.op.tag
    if tensor.op.tag.find("|") != -1:
        insn = tag.split("|")[0]
    else:
        insn = tag
    return insn


def support_scalar(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return get_dsl_insn(tensor) in SUPPORT_SCALAR_INSNS


def is_v100():
    """
    :return:
    """
    return get_soc_spec(SOC_VERSION) in (ASCEND_910, ASCEND_310)


def is_v200():
    """
    :return:
    """
    return get_soc_spec(SOC_VERSION) in (ASCEND_610, ASCEND_615, ASCEND_710, HI3796CV300ES, HI3796CV300CS, SD3403)


def is_v220():
    """
    :return:
    """
    return get_soc_spec(SOC_VERSION) in (ASCEND_920A,)


def need_temp_space(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    op_tag = get_dsl_insn(tensor)
    return op_tag in NEED_TEMP_SPACE_INSNS or \
           (is_v100() and op_tag in NEED_SPACE_WITH_DIFF_TYPE and tensor.dtype == "int32")


def need_extent_node(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return get_dsl_insn(tensor) in NEED_EXTENT_NODE_INSNS


def is_vcmp_insn(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return get_dsl_insn(tensor) in VCMP_INSNS


def is_vsel_insn(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return get_dsl_insn(tensor) == VSEL_INSNS


def is_vcmpsel_insn(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return get_dsl_insn(tensor) in VCMPSEL_INSNS


def get_tensor_size(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    shape = shape_to_list(tensor.shape)
    if all(isinstance(i, int) for i in shape):
        return reduce(mul, shape_to_list(tensor.shape), 1)
    return -1


def is_vtranspose_broadcast(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    if not is_broadcast(tensor) or len(tensor.op.input_tensors) != 1:
        return False
    dtype_no_fp16 = tensor.dtype != "float16"
    compile_broadcast_no_last = is_unified_broadcast(tensor) and \
                                expr_equal(tensor.shape[-1], tensor.op.input_tensors[0].shape[-1]) and \
                                not expr_equal(tensor.shape[-1], 1)
    runtime_broadcast_no_last = expr_equal(tensor.shape[-1], tensor.op.input_tensors[0].shape[-1]) and \
                                isinstance(tensor.shape[-1], tvm.expr.ConstExpr) and \
                                isinstance(tensor.op.input_tensors[0].shape[-1], tvm.expr.ConstExpr) and \
                                not expr_equal(tensor.shape[-1], 1)
    return not (dtype_no_fp16 or compile_broadcast_no_last or runtime_broadcast_no_last)


def is_broadcast(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return get_dsl_insn(tensor) in BROADCAST_INSNS


def is_unknown_broadcast(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return get_dsl_insn(tensor) == "unknown_broadcast"


def is_unified_broadcast(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return get_dsl_insn(tensor) == "unified_broadcast"


def is_scalar_broadcast(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return get_dsl_insn(tensor) == "broadcast"


def is_placeholder(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return isinstance(tensor.op, tvm.tensor.PlaceholderOp)


def merge_value(map_: dict, key, value):
    """
    :param map_:
    :param key:
    :param value: value container is set
    :return:
    """
    if key not in map_:
        map_[key] = set()
    if isinstance(value, list):
        map_[key].update(value)
    else:
        map_[key].add(value)


def get_bound(expr):
    """
    :param expr:
    :return:
    """
    valid_types = (int, tvm.expr.Expr)
    is_true(not isinstance(expr, valid_types),
            {"errCode": "E90001",
            "detailed_cause": "Only accept (int, expr), but now " \
                              "is [%s]." % type(expr)
            })

    if isinstance(expr, int):
        return expr, expr
    if isinstance(expr, tvm.expr.IntImm):
        return expr.value, expr.value
    if isinstance(expr, tvm.expr.Var):
        return operation.get_te_var(expr.name).get_bound()

    def _mul(_a, _b):
        if _a is None or _b is None:
            return None
        _bound = _a * _b
        return None if _bound > VAR_BOUND_LIMIT else _bound

    def _max(_a, _b):
        if _a is None or _b is None:
            return None
        return max(_a, _b)

    def _min(_a, _b):
        if _a is None or _b is None:
            return None
        return min(_a, _b)

    def _parse(_expr):
        if isinstance(_expr, tvm.expr.ConstExpr):
            return _expr.value, _expr.value
        elif isinstance(_expr, tvm.expr.Var):
            bound = operation.get_te_var(_expr.name).get_bound()
            return bound[0], bound[1]
        elif isinstance(_expr, tvm.expr.Mul):
            left_lower, left_upper = _parse(_expr.a)
            right_lower, right_upper = _parse(_expr.b)
            _lower, _upper = _mul(left_lower, right_lower), _mul(left_upper, right_upper)
        elif isinstance(_expr, tvm.expr.Max):
            left_lower, left_upper = _parse(_expr.a)
            right_lower, right_upper = _parse(_expr.b)
            _lower, _upper = _min(left_lower, right_lower), _max(left_upper, right_upper)
        else:
            dict_args = {}
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "Only accept (ConstExpr, Var, Mul, Max), but now " \
                                          "is [%s]" % type(_expr)
            raise RuntimeError(dict_args, get_error_message(dict_args))
        return _lower, _upper

    return _parse(expr)


def get_ub_size():
    """
    :return:
    """
    return get_soc_spec("UB_SIZE")


def get_l1_size():
    """
    :return:
    """
    return get_soc_spec("L1_SIZE")


def get_core_num():
    """
    :return:
    """
    return get_soc_spec("CORE_NUM")


def equals_one(_x):
    """
    :param _x:
    :return:
    """
    if isinstance(_x, tvm.expr.ConstExpr):
        return _x.value == 1
    if isinstance(_x, int):
        return _x == 1
    return False


def ceil_div(num, factor):
    """
    :param num:
    :param factor:
    :return:
    """
    return (num + factor - 1) // factor


def ceil_align(num, factor):
    """
    :param num:
    :param factor:
    :return:
    """
    return ceil_div(num, factor) * factor


def add_sch_additional_entry(sch, k, v):
    """
    :param sch:
    :param k:
    :param v:
    :return:
    """
    if not hasattr(sch, "addition"):
        sch.addition = {}
    sch.addition[k] = v


def get_sch_additional_entry(sch, k):
    """
    :param sch:
    :param k:
    :return:
    """
    if not hasattr(sch, "addition"):
        return None
    return sch.addition.get(k)


def is_reduce_tensor(tensor: Tensor) -> bool:
    """
    Check if tensor contains reduce body
    """
    if isinstance(tensor.op, PlaceholderOp):
        return False
    if isinstance(tensor.op.body[0], Reduce):
        return True
    return False


def get_reduce_axes(reduce_tensor: Tensor) -> List[Union[Var, tvm.expr.IntImm]]:
    """
    Get reduce axes var of reduce tensor
    """
    compute = operation.get_context().get_current_compute()
    if compute.get("_mode") == "zero":
        shape = compute.get("_shape")
        if shape == (1, -1, 0):
            return [tvm.expr.IntImm("int32", 0)]

    if not is_reduce_tensor(reduce_tensor):
        raise RuntimeError("Cannot get reduce axes of non-reduce tensor!")
    reduce_tensor_body = reduce_tensor.op.body
    reduce_tensor_axes = list(reduce_tensor_body[0].axis)
    for idx, axis in enumerate(reduce_tensor_axes):
        reduce_tensor_axes[idx] = axis.var
    return reduce_tensor_axes


def get_reduce_all_axes(reduce_tensor: Tensor) -> List[Union[Var, tvm.expr.IntImm]]:
    """
    Get all axes var for reduce tensor
    """
    compute = operation.get_context().get_current_compute()
    if compute.get("_mode") == "zero" and compute.get("_shape") == (1, -1, 0):
        return list(reduce_tensor.shape) + [tvm.expr.IntImm("int32", 0)]

    reduce_tensor_body = reduce_tensor.op.body
    return list(reduce_tensor_body[0].source[0].args)


def get_reduce_axis_indexes(reduce_tensor: Tensor) -> List[int]:
    """
    Get all reduce axis index
    """
    compute = operation.get_context().get_current_compute()
    if compute.get("_mode") == "zero" and compute.get("_shape") == (1, -1, 0):
        return [2]

    return [get_reduce_all_axes(reduce_tensor).index(axis) for axis in get_reduce_axes(reduce_tensor)]


def is_keepdims(reduce_tensor: Tensor) -> bool:
    """
    Check if reduce tensor is keepdims
    """
    return len(reduce_tensor.shape) == len(get_reduce_all_axes(reduce_tensor))
